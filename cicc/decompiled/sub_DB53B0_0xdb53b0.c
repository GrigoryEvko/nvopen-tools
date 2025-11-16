// Function: sub_DB53B0
// Address: 0xdb53b0
//
_QWORD *__fastcall sub_DB53B0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r13
  bool v7; // cc
  int v9; // [rsp+8h] [rbp-68h]
  __int64 v10; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-58h]
  const void *v12; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-48h]
  __int64 v14; // [rsp+30h] [rbp-40h] BYREF
  int v15; // [rsp+38h] [rbp-38h]

  sub_DB4FC0((__int64)a1, *a2, **(_QWORD **)(a3 + 32));
  v4 = *(_QWORD *)(a3 + 40);
  if ( (unsigned int)v4 > 1 )
  {
    v5 = 8;
    v6 = 8LL * (unsigned int)v4;
    do
    {
      if ( *((_DWORD *)a1 + 2) <= 0x40u )
      {
        if ( *a1 == 1 )
          return a1;
      }
      else
      {
        v9 = *((_DWORD *)a1 + 2);
        if ( v9 - (unsigned int)sub_C444A0((__int64)a1) <= 0x40 && *(_QWORD *)*a1 == 1 )
          return a1;
      }
      sub_DB4FC0((__int64)&v10, *a2, *(_QWORD *)(*(_QWORD *)(a3 + 32) + v5));
      v13 = *((_DWORD *)a1 + 2);
      if ( v13 > 0x40 )
        sub_C43780((__int64)&v12, (const void **)a1);
      else
        v12 = (const void *)*a1;
      sub_C49E90((__int64)&v14, (__int64)&v12, (__int64)&v10);
      if ( *((_DWORD *)a1 + 2) > 0x40u && *a1 )
        j_j___libc_free_0_0(*a1);
      v7 = v13 <= 0x40;
      *a1 = v14;
      *((_DWORD *)a1 + 2) = v15;
      if ( !v7 && v12 )
        j_j___libc_free_0_0(v12);
      if ( v11 > 0x40 && v10 )
        j_j___libc_free_0_0(v10);
      v5 += 8;
    }
    while ( v5 != v6 );
  }
  return a1;
}
