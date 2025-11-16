// Function: sub_2E971F0
// Address: 0x2e971f0
//
__int64 __fastcall sub_2E971F0(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // r13d
  unsigned int v4; // eax
  __int64 v5; // rbx
  unsigned __int64 v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  __int128 v9; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h]
  unsigned __int64 v13; // [rsp+28h] [rbp-38h]
  __int64 v14; // [rsp+30h] [rbp-30h]
  __int64 v15; // [rsp+38h] [rbp-28h]

  v2 = 0;
  if ( (unsigned __int8)sub_BB98D0(a1, *a2) )
    return v2;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v9 = 0;
  v4 = sub_2E95AB0(&v9, (__int64)a2);
  v5 = v14;
  v6 = v13;
  v2 = v4;
  if ( v14 != v13 )
  {
    do
    {
      if ( (*(_BYTE *)(v6 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v6 + 16), 16LL * *(unsigned int *)(v6 + 24), 8);
      v6 += 80LL;
    }
    while ( v5 != v6 );
    v6 = v13;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = v11;
  v8 = v10;
  if ( v11 != v10 )
  {
    do
    {
      if ( (*(_BYTE *)(v8 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v8 + 16), 16LL * *(unsigned int *)(v8 + 24), 8);
      v8 += 80LL;
    }
    while ( v7 != v8 );
    v8 = v10;
  }
  if ( !v8 )
    return v2;
  j_j___libc_free_0(v8);
  return v2;
}
