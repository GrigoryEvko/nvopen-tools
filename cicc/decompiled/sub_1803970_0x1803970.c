// Function: sub_1803970
// Address: 0x1803970
//
void __fastcall sub_1803970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // r12
  const char *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  const char *v13; // r10
  size_t v14; // r8
  const char *v15; // rax
  char *v16; // rdx
  unsigned __int64 v17; // rax
  char *v18; // rax
  size_t v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // rdi
  size_t n; // [rsp+8h] [rbp-78h]
  const char *src; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  const char *v27; // [rsp+28h] [rbp-58h] BYREF
  char *v28; // [rsp+30h] [rbp-50h] BYREF
  const char *v29; // [rsp+38h] [rbp-48h]
  _QWORD v30[8]; // [rsp+40h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 48);
  if ( v6 )
    goto LABEL_2;
  v7 = *(_QWORD *)(a2 + 40);
  if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
  {
    v25 = a4;
    v28 = "___asan_gen_";
    v29 = "_anon_global";
    LOWORD(v30[0]) = 771;
    sub_164B780(a2, (__int64 *)&v28);
    a4 = v25;
  }
  if ( a5 && (*(_BYTE *)(a2 + 32) & 0xFu) - 7 <= 1 )
  {
    v24 = a4;
    v10 = sub_1649960(a2);
    v12 = v24;
    v13 = v10;
    v14 = v11;
    if ( !v10 )
    {
      LOBYTE(v30[0]) = 0;
      v17 = 0x3FFFFFFFFFFFFFFFLL;
      v28 = (char *)v30;
      v29 = 0;
      goto LABEL_18;
    }
    v27 = (const char *)v11;
    v15 = (const char *)v11;
    v28 = (char *)v30;
    if ( v11 > 0xF )
    {
      n = v11;
      src = v13;
      v20 = (_QWORD *)sub_22409D0(&v28, &v27, 0);
      v12 = v24;
      v13 = src;
      v21 = v20;
      v14 = n;
      v28 = (char *)v20;
      v30[0] = v27;
    }
    else
    {
      if ( v11 == 1 )
      {
        LOBYTE(v30[0]) = *v13;
        v16 = (char *)v30;
LABEL_11:
        v29 = v15;
        v15[(_QWORD)v16] = 0;
        v17 = 0x3FFFFFFFFFFFFFFFLL - (_QWORD)v29;
LABEL_18:
        if ( a5 > v17 )
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490(&v28, v12, a5, v12);
        v6 = sub_1633B90(v7, v28, (size_t)v29);
        if ( v28 != (char *)v30 )
          j_j___libc_free_0(v28, v30[0] + 1LL);
        goto LABEL_13;
      }
      if ( !v11 )
      {
        v16 = (char *)v30;
        goto LABEL_11;
      }
      v21 = v30;
    }
    v26 = v12;
    memcpy(v21, v13, v14);
    v15 = v27;
    v16 = v28;
    v12 = v26;
    goto LABEL_11;
  }
  v18 = (char *)sub_1649960(a2);
  v6 = sub_1633B90(v7, v18, v19);
LABEL_13:
  if ( *(_DWORD *)(a1 + 276) == 1 )
  {
    *(_DWORD *)(v6 + 8) = 3;
    if ( (*(_BYTE *)(a2 + 32) & 0xF) == 8 )
      *(_WORD *)(a2 + 32) = *(_WORD *)(a2 + 32) & 0xBFC0 | 0x4007;
  }
  *(_QWORD *)(a2 + 48) = v6;
LABEL_2:
  *(_QWORD *)(a3 + 48) = v6;
}
