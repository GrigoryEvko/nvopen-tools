// Function: sub_15A04A0
// Address: 0x15a04a0
//
__int64 __fastcall sub_15A04A0(_QWORD **a1)
{
  char v2; // r12
  unsigned int v3; // ecx
  __int64 v4; // r12
  unsigned int v6; // eax
  __int64 *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rbx
  unsigned __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h] BYREF
  __int64 v17; // [rsp+10h] [rbp-30h]

  v2 = *((_BYTE *)a1 + 8);
  if ( v2 == 11 )
  {
    v3 = *((_DWORD *)a1 + 2) >> 8;
    LODWORD(v16) = v3;
    if ( v3 <= 0x40 )
      v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
    else
      sub_16A4EF0(&v15, -1, 1);
    v4 = sub_159C0E0(*a1, (__int64)&v15);
    if ( (unsigned int)v16 > 0x40 )
    {
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    return v4;
  }
  if ( (unsigned __int8)(v2 - 1) > 5u )
  {
    v11 = sub_15A04A0(a1[3]);
    return sub_15A0390((size_t)a1[4], v11);
  }
  else
  {
    v6 = sub_1643030(a1);
    sub_169D1B0(&v15, v6, v2 != 6);
    v7 = *a1;
    v4 = sub_159CCF0(*a1, (__int64)&v15);
    v10 = sub_16982C0(v7, &v15, v8, v9);
    if ( v16 == v10 )
    {
      v12 = v17;
      if ( v17 )
      {
        v13 = 32LL * *(_QWORD *)(v17 - 8);
        v14 = v17 + v13;
        if ( v17 != v17 + v13 )
        {
          do
          {
            v14 -= 32;
            sub_127D120((_QWORD *)(v14 + 8));
          }
          while ( v12 != v14 );
        }
        j_j_j___libc_free_0_0(v12 - 8);
      }
      return v4;
    }
    sub_1698460(&v16);
    return v4;
  }
}
