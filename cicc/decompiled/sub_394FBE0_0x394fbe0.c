// Function: sub_394FBE0
// Address: 0x394fbe0
//
__int64 __fastcall sub_394FBE0(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // [rsp+0h] [rbp-70h] BYREF
  __int16 v8; // [rsp+10h] [rbp-60h]
  _QWORD *v9; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v10[3]; // [rsp+30h] [rbp-40h] BYREF
  int v11; // [rsp+4Ch] [rbp-24h]

  v8 = 260;
  v7 = sub_15F2050(a1) + 240;
  sub_16E1010((__int64)&v9, (__int64)&v7);
  if ( v11 == 7 || v11 == 29 )
  {
    if ( v9 != v10 )
    {
      j_j___libc_free_0((unsigned __int64)v9);
      return 0;
    }
    return 0;
  }
  if ( v9 != v10 )
    j_j___libc_free_0((unsigned __int64)v9);
  v3 = *(_QWORD *)(a1 + 64);
  v4 = *(_QWORD *)(v3 + 16);
  LOBYTE(v1) = *(_BYTE *)(*(_QWORD *)v4 + 8LL) != 0 && (*(_BYTE *)(*(_QWORD *)v4 + 8LL) & 0xFB) != 11;
  if ( (_BYTE)v1 )
    return 0;
  v5 = v4 + 8;
  v6 = v4 + 8LL * *(unsigned int *)(v3 + 12);
  if ( v4 + 8 != v6 )
  {
    while ( (*(_BYTE *)(*(_QWORD *)v5 + 8LL) & 0xFB) == 0xB )
    {
      v5 += 8;
      if ( v6 == v5 )
        return 1;
    }
    return v1;
  }
  return 1;
}
