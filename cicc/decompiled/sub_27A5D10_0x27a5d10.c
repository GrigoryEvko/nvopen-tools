// Function: sub_27A5D10
// Address: 0x27a5d10
//
__int64 __fastcall sub_27A5D10(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  __int64 i; // r9
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v20; // rcx
  unsigned __int64 v22; // rdi
  __int64 v25; // [rsp+18h] [rbp-138h]
  __int64 v26; // [rsp+20h] [rbp-130h] BYREF
  unsigned __int64 v27; // [rsp+28h] [rbp-128h]
  char v28; // [rsp+3Ch] [rbp-114h]
  unsigned __int64 v29; // [rsp+80h] [rbp-D0h]
  unsigned __int64 v30; // [rsp+88h] [rbp-C8h]
  __int64 v31; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int64 v32; // [rsp+A8h] [rbp-A8h]
  char v33; // [rsp+BCh] [rbp-94h]
  unsigned __int64 v34; // [rsp+100h] [rbp-50h]
  __int64 v35; // [rsp+108h] [rbp-48h]

  v6 = *(_QWORD *)(a3 + 64);
  v25 = *(_QWORD *)(a2 + 40);
  sub_27A5910(&v26, v6);
  sub_27A1350(&v31, v6, v7, v8, v9, v10);
  v12 = v29;
  v13 = v30;
  v14 = v34;
  for ( i = v35; ; i = v35 )
  {
    while ( 1 )
    {
      v18 = v13 - v12;
      if ( v13 - v12 == i - v14 )
      {
        if ( v12 == v13 )
        {
LABEL_16:
          if ( v14 )
            j_j___libc_free_0(v14);
          if ( !v33 )
            _libc_free(v32);
          if ( v29 )
            j_j___libc_free_0(v29);
          if ( !v28 )
            _libc_free(v27);
          return 0;
        }
        v20 = v14;
        while ( 1 )
        {
          v18 = *(_QWORD *)v20;
          if ( *(_QWORD *)v12 != *(_QWORD *)v20 )
            break;
          v18 = *(unsigned __int8 *)(v12 + 16);
          if ( (_BYTE)v18 != *(_BYTE *)(v20 + 16) )
            break;
          if ( (_BYTE)v18 )
          {
            v18 = *(_QWORD *)(v20 + 8);
            if ( *(_QWORD *)(v12 + 8) != v18 )
              break;
          }
          v12 += 24;
          v20 += 24LL;
          if ( v12 == v13 )
            goto LABEL_16;
        }
      }
      v16 = *(_QWORD *)(v13 - 24);
      if ( v25 != v16 )
        break;
      v19 = v13 - 24;
      v13 = v29;
      v30 = v19;
      v12 = v29;
      if ( v19 != v29 )
        goto LABEL_7;
    }
    if ( (unsigned __int8)sub_27A56C0(a1, *(_QWORD *)(v13 - 24), v6, a4) )
      break;
    v18 = a2;
    if ( (unsigned __int8)sub_27A2490(a1, a2, a3, v16, v17) )
      break;
    if ( *a4 != -1 )
      --*a4;
LABEL_7:
    sub_27A57B0((__int64)&v26, v18, v12, v19, v11, i);
    v12 = v29;
    v13 = v30;
    v14 = v34;
  }
  if ( v34 )
    j_j___libc_free_0(v34);
  if ( v33 )
  {
    v22 = v29;
    if ( v29 )
      goto LABEL_31;
  }
  else
  {
    _libc_free(v32);
    v22 = v29;
    if ( v29 )
LABEL_31:
      j_j___libc_free_0(v22);
  }
  if ( !v28 )
    _libc_free(v27);
  return 1;
}
