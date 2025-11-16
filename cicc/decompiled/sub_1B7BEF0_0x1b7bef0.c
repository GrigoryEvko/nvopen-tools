// Function: sub_1B7BEF0
// Address: 0x1b7bef0
//
__int64 __fastcall sub_1B7BEF0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 i; // r14
  _QWORD *v12; // r13
  __int64 v13; // rax
  _BYTE *v14; // rbx
  _BYTE *v15; // r14
  int v16; // eax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // r12
  _QWORD *v22; // rax
  _QWORD *v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int8 v26; // cl
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v30; // rax
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // [rsp+8h] [rbp-108h]
  char *v35; // [rsp+10h] [rbp-100h] BYREF
  char v36; // [rsp+20h] [rbp-F0h]
  char v37; // [rsp+21h] [rbp-EFh]
  _BYTE *v38; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-D8h]
  _BYTE v40[208]; // [rsp+40h] [rbp-D0h] BYREF

  v9 = a1 + 72;
  v10 = *(_QWORD *)(a1 + 80);
  v38 = v40;
  v39 = 0x1400000000LL;
  if ( a1 + 72 == v10 )
  {
    i = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v10 + 24);
      if ( i != v10 + 16 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v9 == v10 )
        break;
      if ( !v10 )
        BUG();
    }
  }
  v12 = &v38;
LABEL_8:
  while ( v9 != v10 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) == 78 )
    {
      v30 = *(_QWORD *)(i - 48);
      if ( !*(_BYTE *)(v30 + 16)
        && (*(_BYTE *)(v30 + 33) & 0x20) != 0
        && *(_DWORD *)(v30 + 36) == 76
        && sub_1642D30(*(_QWORD *)(i - 24LL * (*(_DWORD *)(i - 4) & 0xFFFFFFF) - 24)) )
      {
        v33 = (unsigned int)v39;
        if ( (unsigned int)v39 >= HIDWORD(v39) )
        {
          sub_16CD150((__int64)&v38, v40, 0, 8, v31, v32);
          v33 = (unsigned int)v39;
        }
        *(_QWORD *)&v38[8 * v33] = i - 24;
        LODWORD(v39) = v39 + 1;
      }
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v10 + 24) )
    {
      v13 = v10 - 24;
      if ( !v10 )
        v13 = 0;
      if ( i != v13 + 40 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v9 == v10 )
        goto LABEL_8;
      if ( !v10 )
        BUG();
    }
  }
  v14 = v38;
  v15 = &v38[8 * (unsigned int)v39];
  v16 = v39;
  if ( v38 != v15 )
  {
    while ( 1 )
    {
      v12 = *(_QWORD **)v14;
      v24 = *(_DWORD *)(*(_QWORD *)v14 + 20LL) & 0xFFFFFFF;
      v25 = *(_QWORD *)(*(_QWORD *)v14 - 24 * v24);
      v26 = *(_BYTE *)(v25 + 16);
      if ( v26 == 88 )
      {
        v28 = sub_157F120(*(_QWORD *)(v25 + 40));
        v25 = sub_157EBA0(v28);
        v26 = *(_BYTE *)(v25 + 16);
        v24 = *((_DWORD *)v12 + 5) & 0xFFFFFFF;
      }
      if ( v26 > 0x17u )
      {
        if ( v26 == 78 )
        {
          v27 = v25 | 4;
          goto LABEL_36;
        }
        v17 = 0;
        if ( v26 == 29 )
        {
          v27 = v25 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_36:
          v17 = v27 & 0xFFFFFFFFFFFFFFF8LL;
          v18 = (v27 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
          if ( (v27 & 4) != 0 )
            goto LABEL_24;
        }
      }
      else
      {
        v17 = 0;
      }
      v18 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
LABEL_24:
      v19 = v12[3 * (2 - v24)];
      v20 = *(_QWORD **)(v19 + 24);
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
        v20 = (_QWORD *)*v20;
      v21 = *(_QWORD **)(v18 + 24LL * (unsigned int)v20);
      v34 = *v12;
      if ( *v12 != *v21 )
      {
        v37 = 1;
        v35 = "cast";
        v36 = 3;
        v22 = sub_1648A60(56, 1u);
        v23 = v22;
        if ( v22 )
          sub_15FD590((__int64)v22, (__int64)v21, v34, (__int64)&v35, (__int64)v12);
        v21 = v23;
      }
      v14 += 8;
      sub_164D160((__int64)v12, (__int64)v21, a2, a3, a4, a5, a6, a7, a8, a9);
      sub_15F20C0(v12);
      if ( v15 == v14 )
      {
        v16 = v39;
        v15 = v38;
        break;
      }
    }
  }
  LOBYTE(v12) = v16 != 0;
  if ( v15 != v40 )
    _libc_free((unsigned __int64)v15);
  return (unsigned int)v12;
}
