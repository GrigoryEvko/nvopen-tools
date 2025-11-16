// Function: sub_F0C4B0
// Address: 0xf0c4b0
//
__int64 __fastcall sub_F0C4B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdi
  __int64 *v11; // r10
  __int64 v12; // rdx
  __int64 *v13; // [rsp+0h] [rbp-70h]
  const void *v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h]
  __int64 v17; // [rsp+38h] [rbp-38h]

  v9 = *(_QWORD *)(a2 - 32);
  if ( v9 )
  {
    if ( *(_BYTE *)v9 )
    {
      v9 = 0;
    }
    else if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v9 = 0;
    }
  }
  if ( (unsigned __int8)sub_B2DD60(v9) )
  {
    v11 = *(__int64 **)(a1 + 8);
    v15 = *(_DWORD *)(a3 + 8);
    if ( v15 > 0x40 )
    {
      v13 = v11;
      sub_C43780((__int64)&v14, (const void **)a3);
      v11 = v13;
    }
    else
    {
      v14 = *(const void **)a3;
    }
    v16 = sub_DF9D60(v11, a1, a2, (__int64)&v14, a4, a5);
    v17 = v12;
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  else
  {
    LOBYTE(v17) = 0;
  }
  return v16;
}
