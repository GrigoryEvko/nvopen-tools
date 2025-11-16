// Function: sub_256B200
// Address: 0x256b200
//
__int64 __fastcall sub_256B200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r12
  char v6; // r15
  unsigned int v7; // r13d
  __int64 v9; // [rsp+8h] [rbp-58h]
  _BYTE v10[16]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 v11; // [rsp+20h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 88) )
  {
    v5 = *(__int64 **)(a2 + 72);
    v6 = 0;
    v9 = a2 + 56;
  }
  else
  {
    v5 = *(__int64 **)a2;
    v6 = 1;
    v9 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  }
  v7 = 0;
  while ( v6 )
  {
    if ( (__int64 *)v9 == v5 )
      return v7;
    sub_256AFA0((__int64)v10, a1, v5, a4, a5);
    if ( v11 )
      v7 = v11;
    ++v5;
  }
  while ( (__int64 *)v9 != v5 )
  {
    sub_256AFA0((__int64)v10, a1, v5 + 4, a4, a5);
    if ( v11 )
      v7 = v11;
    v5 = (__int64 *)sub_220EF30((__int64)v5);
  }
  return v7;
}
