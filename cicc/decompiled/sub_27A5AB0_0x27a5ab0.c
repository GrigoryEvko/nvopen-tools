// Function: sub_27A5AB0
// Address: 0x27a5ab0
//
__int64 __fastcall sub_27A5AB0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r9
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __int64 i; // r8
  __int64 v18; // rsi
  __int64 result; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rcx
  char v22; // si
  unsigned __int8 v23; // [rsp+Fh] [rbp-131h]
  unsigned __int8 v24; // [rsp+Fh] [rbp-131h]
  unsigned __int8 v25; // [rsp+Fh] [rbp-131h]
  unsigned __int8 v26; // [rsp+Fh] [rbp-131h]
  __int64 v27; // [rsp+10h] [rbp-130h] BYREF
  unsigned __int64 v28; // [rsp+18h] [rbp-128h]
  char v29; // [rsp+2Ch] [rbp-114h]
  unsigned __int64 v30; // [rsp+70h] [rbp-D0h]
  unsigned __int64 v31; // [rsp+78h] [rbp-C8h]
  __int64 v32; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 v33; // [rsp+98h] [rbp-A8h]
  char v34; // [rsp+ACh] [rbp-94h]
  unsigned __int64 v35; // [rsp+F0h] [rbp-50h]
  __int64 v36; // [rsp+F8h] [rbp-48h]

  sub_27A5910(&v27, a3);
  sub_27A1350(&v32, a3, v9, v10, v11, v12);
  v14 = v30;
  v15 = v31;
  v16 = v35;
  for ( i = v36; ; i = v36 )
  {
    while ( 1 )
    {
      if ( v15 - v14 == i - v16 )
      {
        if ( v15 == v14 )
        {
LABEL_15:
          if ( v16 )
            j_j___libc_free_0(v16);
          if ( !v34 )
            _libc_free(v33);
          if ( v30 )
            j_j___libc_free_0(v30);
          if ( !v29 )
            _libc_free(v28);
          return 0;
        }
        v21 = v16;
        while ( *(_QWORD *)v14 == *(_QWORD *)v21 )
        {
          v22 = *(_BYTE *)(v14 + 16);
          if ( v22 != *(_BYTE *)(v21 + 16) || v22 && *(_QWORD *)(v14 + 8) != *(_QWORD *)(v21 + 8) )
            break;
          v14 += 24;
          v21 += 24LL;
          if ( v15 == v14 )
            goto LABEL_15;
        }
      }
      v18 = *(_QWORD *)(v15 - 24);
      if ( v18 != a2 )
        break;
      v20 = v15 - 24;
      v15 = v30;
      v31 = v20;
      v14 = v30;
      if ( v20 != v30 )
        goto LABEL_6;
    }
    result = sub_27A56C0(a1, v18, a3, a4);
    if ( (_BYTE)result )
      break;
    if ( *a4 != -1 )
      --*a4;
LABEL_6:
    sub_27A57B0((__int64)&v27, v18, v14, v20, i, v13);
    v14 = v30;
    v15 = v31;
    v16 = v35;
  }
  if ( v35 )
  {
    v23 = result;
    j_j___libc_free_0(v35);
    result = v23;
  }
  if ( !v34 )
  {
    v26 = result;
    _libc_free(v33);
    result = v26;
  }
  if ( v30 )
  {
    v24 = result;
    j_j___libc_free_0(v30);
    result = v24;
  }
  if ( !v29 )
  {
    v25 = result;
    _libc_free(v28);
    return v25;
  }
  return result;
}
