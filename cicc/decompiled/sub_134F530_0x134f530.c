// Function: sub_134F530
// Address: 0x134f530
//
__int64 __fastcall sub_134F530(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  int v7; // eax
  char v8; // bl
  int v9; // eax
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rbx
  int v12; // r13d
  char v13; // al
  char v14; // cl
  unsigned __int8 v15; // al
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // r8
  char v18; // r10
  int v19; // r14d
  unsigned __int64 v20; // r13
  int v21; // r13d
  unsigned __int8 v22; // al
  char v23; // [rsp+8h] [rbp-88h]
  unsigned __int64 v24; // [rsp+8h] [rbp-88h]
  unsigned __int64 v25; // [rsp+10h] [rbp-80h]
  char v26; // [rsp+10h] [rbp-80h]
  char v27; // [rsp+18h] [rbp-78h]
  __int64 i; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v30[96]; // [rsp+30h] [rbp-60h] BYREF

  v3 = 7;
  v4 = (_QWORD *)a1[6];
  v5 = (_QWORD *)a1[7];
  v29 = a2;
  for ( i = a3; v5 != v4; ++v4 )
  {
    v3 &= (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v4 + 72LL))(*v4, v29, i);
    if ( (v3 & 3) == 0 )
      return 4;
  }
  v7 = sub_134CC90((__int64)a1, v29);
  v8 = v7;
  if ( v7 == 4 )
    return 4;
  v9 = sub_134CC90((__int64)a1, i);
  if ( v9 == 4 )
    return 4;
  if ( (v8 & 2) != 0 )
  {
    if ( (v8 & 1) == 0 )
      v3 &= 6u;
    if ( (v9 & 0x30) == 0 )
      goto LABEL_12;
LABEL_27:
    if ( (v8 & 0x30) != 0 )
      return v3;
    if ( (v8 & 3) == 0 )
      return 4;
    if ( (v8 & 8) == 0 )
      return 4;
    v16 = (v29 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
    v17 = sub_134EF80(&v29);
    if ( v17 == v16 )
      return 4;
    v18 = 1;
    v19 = 4;
    while ( 1 )
    {
      if ( *(_BYTE *)(**(_QWORD **)v16 + 8LL) == 15 )
      {
        v24 = v17;
        v26 = v18;
        v20 = 0xAAAAAAAAAAAAAAABLL
            * ((__int64)(v16
                       + 24LL * (*(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                       - (v29 & 0xFFFFFFFFFFFFFFF8LL)) >> 3);
        sub_141F820(
          v30,
          v29,
          -1431655765
        * (unsigned int)((__int64)(v16
                                 + 24LL * (*(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                                 - (v29 & 0xFFFFFFFFFFFFFFF8LL)) >> 3),
          a1[5]);
        v21 = sub_134CC10((__int64)a1, v29, v20);
        v22 = sub_134F0E0(a1, i, (__int64)v30);
        v17 = v24;
        if ( (v21 & 2) != 0 && (v22 & 3) != 0 || (v21 & 1) != 0 && (v22 & 2) != 0 )
          v19 = v3 & (v19 | v21);
        v18 = ((v22 >> 2) ^ 1) & v26;
        if ( (_BYTE)v3 == (_BYTE)v19 )
          break;
      }
      v16 += 24LL;
      if ( v17 == v16 )
        goto LABEL_47;
    }
    if ( v24 != v16 + 24 )
    {
      if ( (v19 & 3) == 0 )
        return 4;
      return v19 | 4u;
    }
LABEL_47:
    v3 = v19 & 3;
    if ( (v19 & 3) != 0 )
    {
      if ( v18 )
        return v3;
      return v19 | 4u;
    }
    return 4;
  }
  v3 &= 5u;
  if ( (v9 & 2) == 0 )
    return 4;
  if ( (v9 & 0x30) != 0 )
    goto LABEL_27;
LABEL_12:
  if ( (v9 & 3) == 0 )
    return 4;
  if ( (v9 & 8) == 0 )
    return 4;
  v10 = (i & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v11 = sub_134EF80(&i);
  if ( v11 == v10 )
    return 4;
  v23 = 1;
  v12 = 4;
  while ( 1 )
  {
    if ( *(_BYTE *)(**(_QWORD **)v10 + 8LL) == 15 )
    {
      v25 = 0xAAAAAAAAAAAAAAABLL
          * ((__int64)(v10
                     + 24LL * (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                     - (i & 0xFFFFFFFFFFFFFFF8LL)) >> 3);
      sub_141F820(
        v30,
        i,
        -1431655765
      * (unsigned int)((__int64)(v10
                               + 24LL * (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                               - (i & 0xFFFFFFFFFFFFFFF8LL)) >> 3),
        a1[5]);
      v13 = sub_134CC10((__int64)a1, i, v25);
      v14 = 7;
      if ( (v13 & 2) == 0 )
        v14 = (v13 & 1) == 0 ? 4 : 6;
      v27 = v14;
      v15 = sub_134F0E0(a1, v29, (__int64)v30);
      v23 &= (v15 >> 2) ^ 1;
      v12 = v3 & ((unsigned __int8)(v27 & v15) | v12);
      if ( (_BYTE)v3 == (_BYTE)v12 )
        break;
    }
    v10 += 24LL;
    if ( v11 == v10 )
      goto LABEL_44;
  }
  if ( v11 != v10 + 24 )
  {
    if ( (v12 & 3) == 0 )
      return 4;
    return v12 | 4u;
  }
LABEL_44:
  v3 = v12 & 3;
  if ( (v12 & 3) == 0 )
    return 4;
  if ( !v23 )
    return v12 | 4u;
  return v3;
}
