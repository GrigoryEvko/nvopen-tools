// Function: sub_3245420
// Address: 0x3245420
//
__int64 __fastcall sub_3245420(__int64 a1, __int64 a2, char *a3, __int64 a4, __int64 a5)
{
  char *v5; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  char **v18; // rsi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r8
  __int64 result; // rax
  char **v22; // rdi
  __int64 v23; // rdx
  _BYTE *v24; // rdi
  __int64 v25; // r9
  __int64 v26; // r9
  char *v27; // r12
  _QWORD *v28; // r9
  _QWORD *v29; // rax
  __int64 v30; // [rsp+0h] [rbp-B0h]
  unsigned int v31; // [rsp+0h] [rbp-B0h]
  unsigned int v32; // [rsp+8h] [rbp-A8h]
  __int64 v33; // [rsp+8h] [rbp-A8h]
  char *v34; // [rsp+10h] [rbp-A0h] BYREF
  char v35; // [rsp+30h] [rbp-80h]
  char v36; // [rsp+31h] [rbp-7Fh]
  __int64 v37; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38; // [rsp+48h] [rbp-68h]
  _BYTE *v39; // [rsp+50h] [rbp-60h] BYREF
  __int64 v40; // [rsp+58h] [rbp-58h]
  _BYTE dest[80]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a3;
  v7 = *(unsigned int *)(a1 + 248);
  if ( !(_DWORD)v7 )
    goto LABEL_4;
  a3 = *(char **)(a1 + 240);
  a4 = 0x200000000LL;
  v8 = (__int64)&a3[64 * v7 - 64];
  v37 = *(_QWORD *)v8;
  v9 = *(_QWORD *)(v8 + 8);
  v39 = dest;
  v38 = v9;
  v40 = 0x200000000LL;
  a5 = *(unsigned int *)(v8 + 24);
  if ( !(_DWORD)a5 || (a3 = (char *)(v8 + 16), &v39 == (_BYTE **)(v8 + 16)) )
  {
    if ( a2 != v9 || *((_DWORD *)v5 + 2) )
      goto LABEL_4;
    v24 = dest;
    goto LABEL_17;
  }
  v25 = (unsigned int)a5;
  v24 = dest;
  a3 = (char *)(16LL * (unsigned int)a5);
  if ( (unsigned int)a5 <= 2
    || (v31 = *(_DWORD *)(v8 + 24),
        v33 = (unsigned int)a5,
        sub_C8D5F0((__int64)&v39, dest, (unsigned int)a5, 0x10u, a5, (unsigned int)a5),
        v24 = v39,
        v25 = v33,
        a5 = v31,
        (a3 = (char *)(16LL * *(unsigned int *)(v8 + 24))) != 0) )
  {
    v30 = v25;
    v32 = a5;
    memcpy(v24, *(const void **)(v8 + 16), (size_t)a3);
    v24 = v39;
    v25 = v30;
    a5 = v32;
  }
  LODWORD(v40) = a5;
  if ( v38 == a2 && v25 == *((_DWORD *)v5 + 2) )
  {
    a3 = *(char **)v5;
    v28 = &v24[16 * v25];
    if ( v24 != (_BYTE *)v28 )
    {
      v29 = v24;
      do
      {
        a4 = *(_QWORD *)a3;
        if ( *v29 != *(_QWORD *)a3 )
          goto LABEL_23;
        a4 = *((_QWORD *)a3 + 1);
        if ( v29[1] != a4 )
          goto LABEL_23;
        v29 += 2;
        a3 += 16;
      }
      while ( v28 != v29 );
    }
LABEL_17:
    if ( v24 == dest )
      return (unsigned int)(*(_DWORD *)(a1 + 248) - 1);
    goto LABEL_12;
  }
LABEL_23:
  if ( v24 != dest )
    _libc_free((unsigned __int64)v24);
LABEL_4:
  v10 = *(_QWORD *)a1;
  v36 = 1;
  v34 = "debug_ranges";
  v35 = 3;
  v11 = sub_31DCC50(v10, (__int64 *)&v34, (__int64)a3, a4, a5);
  v15 = *((unsigned int *)v5 + 2);
  v38 = a2;
  v37 = v11;
  v39 = dest;
  v40 = 0x200000000LL;
  if ( (_DWORD)v15 )
    sub_3244890((__int64)&v39, (char **)v5, v12, v15, v13, v14);
  v16 = *(unsigned int *)(a1 + 248);
  v17 = *(unsigned int *)(a1 + 252);
  v18 = (char **)&v37;
  v19 = *(_QWORD *)(a1 + 240);
  v20 = v16 + 1;
  result = v16;
  if ( v16 + 1 > v17 )
  {
    v26 = a1 + 240;
    if ( v19 > (unsigned __int64)&v37 || (unsigned __int64)&v37 >= v19 + (v16 << 6) )
    {
      sub_3245300(a1 + 240, v20, v19, v17, v20, v26);
      v16 = *(unsigned int *)(a1 + 248);
      v19 = *(_QWORD *)(a1 + 240);
      v18 = (char **)&v37;
      result = v16;
    }
    else
    {
      v27 = (char *)&v37 - v19;
      sub_3245300(a1 + 240, v20, v19, v17, v20, v26);
      v19 = *(_QWORD *)(a1 + 240);
      v16 = *(unsigned int *)(a1 + 248);
      v18 = (char **)&v27[v19];
      result = v16;
    }
  }
  v22 = (char **)(v19 + (v16 << 6));
  if ( v22 )
  {
    *v22 = *v18;
    v22[1] = v18[1];
    v22[2] = (char *)(v22 + 4);
    v22[3] = (char *)0x200000000LL;
    v23 = *((unsigned int *)v18 + 6);
    if ( (_DWORD)v23 )
      sub_3244890((__int64)(v22 + 2), v18 + 2, v23, v17, v20, v14);
    result = *(unsigned int *)(a1 + 248);
  }
  v24 = v39;
  *(_DWORD *)(a1 + 248) = result + 1;
  if ( v24 != dest )
  {
LABEL_12:
    _libc_free((unsigned __int64)v24);
    return (unsigned int)(*(_DWORD *)(a1 + 248) - 1);
  }
  return result;
}
