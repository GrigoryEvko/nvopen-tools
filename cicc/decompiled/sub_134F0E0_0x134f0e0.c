// Function: sub_134F0E0
// Address: 0x134f0e0
//
__int64 __fastcall sub_134F0E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned int v5; // r12d
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  int v9; // eax
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rax
  char v12; // bl
  unsigned __int64 v13; // r14
  char v14; // al
  char v15; // cl
  char v16; // al
  unsigned __int8 v17; // [rsp+5h] [rbp-7Bh]
  char v18; // [rsp+6h] [rbp-7Ah]
  char v19; // [rsp+7h] [rbp-79h]
  unsigned __int64 v20; // [rsp+8h] [rbp-78h]
  char v21; // [rsp+10h] [rbp-70h]
  __int64 i; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v23[96]; // [rsp+20h] [rbp-60h] BYREF

  v3 = (__int64)a1;
  v5 = 7;
  v6 = (_QWORD *)a1[6];
  v7 = (_QWORD *)a1[7];
  for ( i = a2; v7 != v6; ++v6 )
  {
    v5 &= (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(*(_QWORD *)*v6 + 64LL))(*v6, i, a3);
    if ( (v5 & 3) == 0 )
      return 4;
  }
  v9 = sub_134CC90((__int64)a1, i);
  if ( v9 == 23 || v9 == 4 )
    return 4;
  if ( (v9 & 2) != 0 )
  {
    if ( (v9 & 1) == 0 )
      v5 &= 6u;
  }
  else
  {
    v5 &= 5u;
  }
  if ( (v9 & 0x30) != 0 && (v9 & 0x20) != 0 )
    goto LABEL_24;
  if ( (v9 & 3) == 0 )
    return 4;
  if ( (v9 & 8) == 0 )
    return 4;
  v10 = (i & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v11 = sub_134EF80(&i);
  if ( v11 == v10 )
    return 4;
  v18 = 0;
  v17 = v5;
  v12 = 1;
  v13 = v11;
  v19 = 4;
  do
  {
    if ( *(_BYTE *)(**(_QWORD **)v10 + 8LL) == 15 )
    {
      v20 = 0xAAAAAAAAAAAAAAABLL
          * ((__int64)(v10
                     + 24LL * (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                     - (i & 0xFFFFFFFFFFFFFFF8LL)) >> 3);
      sub_141F820(
        v23,
        i,
        -1431655765
      * (unsigned int)((__int64)(v10
                               + 24LL * (*(_DWORD *)((i & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                               - (i & 0xFFFFFFFFFFFFFFF8LL)) >> 3),
        a1[5]);
      v14 = sub_134CB50((__int64)a1, (__int64)v23, a3);
      v15 = v14;
      if ( v14 )
      {
        v21 = v14;
        v16 = sub_134CC10((__int64)a1, i, v20);
        v18 = 1;
        v15 = v21;
        v19 |= v16;
      }
      v12 &= v15 == 3;
    }
    v10 += 24LL;
  }
  while ( v13 != v10 );
  v3 = (__int64)a1;
  v5 = (unsigned __int8)v5;
  if ( !v18 )
    return 4;
  LOBYTE(v5) = v19 & v5;
  v5 |= 4u;
  if ( v12 )
    v5 = (unsigned __int8)v19 & v17 & 3;
LABEL_24:
  if ( (v5 & 2) != 0 && (unsigned __int8)sub_134CBB0(v3, a3, 0) )
    return v5 & 5;
  return v5;
}
