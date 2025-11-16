// Function: sub_21DDC40
// Address: 0x21ddc40
//
__int64 __fastcall sub_21DDC40(__int64 *a1, unsigned int a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rax
  const char *v7; // rax
  size_t v8; // rdx
  _WORD *v9; // rdi
  char *v10; // rsi
  unsigned __int64 v11; // rax
  void **v12; // r15
  __int64 *v13; // rbx
  __int64 v14; // r15
  __int64 *v15; // r13
  int v16; // eax
  __int64 v17; // rax
  unsigned int v18; // r12d
  __int64 v20; // r15
  __int64 *v21; // rbx
  __int64 *v22; // r13
  __int64 v23; // rdi
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v28; // [rsp+18h] [rbp-B8h]
  _QWORD *v29; // [rsp+20h] [rbp-B0h]
  __int64 *v30; // [rsp+28h] [rbp-A8h]
  size_t v31; // [rsp+28h] [rbp-A8h]
  __int64 v32; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v33; // [rsp+38h] [rbp-98h]
  __int64 v34; // [rsp+40h] [rbp-90h]
  __int64 v35; // [rsp+48h] [rbp-88h]
  char *v36[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v37[2]; // [rsp+60h] [rbp-70h] BYREF
  void *v38; // [rsp+70h] [rbp-60h] BYREF
  void *v39; // [rsp+78h] [rbp-58h]
  __int64 v40; // [rsp+80h] [rbp-50h]
  void *dest; // [rsp+88h] [rbp-48h]
  int v42; // [rsp+90h] [rbp-40h]
  char **v43; // [rsp+98h] [rbp-38h]

  v6 = a1[5];
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v28 = v6;
  LOBYTE(v37[0]) = 0;
  v29 = sub_21DC2B0((__int64)a1);
  v36[0] = (char *)v37;
  v36[1] = 0;
  v42 = 1;
  dest = 0;
  v40 = 0;
  v39 = 0;
  v38 = &unk_49EFBE0;
  v43 = v36;
  v7 = sub_1E0A440(a1);
  v9 = dest;
  v10 = (char *)v7;
  v11 = v40 - (_QWORD)dest;
  if ( v8 > v40 - (__int64)dest )
  {
    v25 = sub_16E7EE0((__int64)&v38, v10, v8);
    v9 = *(_WORD **)(v25 + 24);
    v12 = (void **)v25;
    v11 = *(_QWORD *)(v25 + 16) - (_QWORD)v9;
  }
  else
  {
    v12 = &v38;
    if ( v8 )
    {
      v31 = v8;
      memcpy(dest, v10, v8);
      v24 = v40 - ((_QWORD)dest + v31);
      dest = (char *)dest + v31;
      v9 = dest;
      if ( v24 > 6 )
        goto LABEL_4;
      goto LABEL_41;
    }
  }
  if ( v11 > 6 )
  {
LABEL_4:
    *(_DWORD *)v9 = 1918988383;
    v9[2] = 28001;
    *((_BYTE *)v9 + 6) = 95;
    v12[3] = (char *)v12[3] + 7;
    goto LABEL_5;
  }
LABEL_41:
  v12 = (void **)sub_16E7EE0((__int64)v12, "_param_", 7u);
LABEL_5:
  sub_16E7A90((__int64)v12, a2);
  if ( dest != v39 )
    sub_16E7BA0((__int64 *)&v38);
  v13 = (__int64 *)a1[41];
  v30 = a1 + 40;
  if ( v13 == a1 + 40 )
  {
LABEL_17:
    v18 = 0;
    sub_21DC560((__int64)v29, v36[0]);
  }
  else
  {
    while ( 1 )
    {
      v14 = v13[4];
      v15 = v13 + 3;
      if ( (__int64 *)v14 != v13 + 3 )
        break;
LABEL_16:
      v13 = (__int64 *)v13[1];
      if ( v30 == v13 )
        goto LABEL_17;
    }
    while ( 1 )
    {
      v16 = **(unsigned __int16 **)(v14 + 16);
      if ( v16 == 3060 || v16 == 3066 )
      {
        v17 = *(_QWORD *)(v14 + 32);
        if ( *(_BYTE *)(v17 + 240) == 9 && !(unsigned int)sub_2241AC0(v36, *(_QWORD *)(v17 + 264)) )
          break;
      }
      if ( (*(_BYTE *)v14 & 4) != 0 )
      {
        v14 = *(_QWORD *)(v14 + 8);
        if ( v15 == (__int64 *)v14 )
          goto LABEL_16;
      }
      else
      {
        while ( (*(_BYTE *)(v14 + 46) & 8) != 0 )
          v14 = *(_QWORD *)(v14 + 8);
        v14 = *(_QWORD *)(v14 + 8);
        if ( v15 == (__int64 *)v14 )
          goto LABEL_16;
      }
    }
    v18 = sub_21DD1A0(v14, a3, (__int64)&v32, a4, v28);
    if ( (_BYTE)v18 )
    {
      v20 = (unsigned int)sub_21DC560((__int64)v29, v36[0]);
      if ( (_DWORD)v34 )
      {
        v21 = v33;
        v22 = &v33[(unsigned int)v35];
        if ( v33 != v22 )
        {
          while ( *v21 == -8 || *v21 == -16 )
          {
            if ( v22 == ++v21 )
              goto LABEL_18;
          }
          while ( v21 != v22 )
          {
            v23 = *v21++;
            sub_1E313C0(v23, v20);
            if ( v21 == v22 )
              break;
            while ( *v21 == -8 || *v21 == -16 )
            {
              if ( v22 == ++v21 )
                goto LABEL_18;
            }
          }
        }
      }
    }
  }
LABEL_18:
  sub_16E7BC0((__int64 *)&v38);
  if ( (_QWORD *)v36[0] != v37 )
    j_j___libc_free_0(v36[0], v37[0] + 1LL);
  j___libc_free_0(v33);
  return v18;
}
