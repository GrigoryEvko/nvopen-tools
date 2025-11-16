// Function: sub_9AC780
// Address: 0x9ac780
//
__int64 __fastcall sub_9AC780(__int64 a1, __int64 *a2, unsigned __int8 a3, __m128i *a4)
{
  unsigned int v4; // r14d
  __int64 v8; // rsi
  __int64 v10; // rdi
  bool v11; // cc
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v15; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-68h]
  __int64 v17; // [rsp+20h] [rbp-60h]
  unsigned int v18; // [rsp+28h] [rbp-58h]
  __int64 v19; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-48h]
  __int64 v21; // [rsp+40h] [rbp-40h]
  unsigned int v22; // [rsp+48h] [rbp-38h]

  v4 = a3;
  v8 = *a2;
  if ( (v8 & 4) == 0 )
  {
    sub_9AC330((__int64)&v19, v8 & 0xFFFFFFFFFFFFFFF8LL, 0, a4);
    if ( *((_DWORD *)a2 + 4) > 0x40u )
    {
      v10 = a2[1];
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    v11 = *((_DWORD *)a2 + 8) <= 0x40u;
    a2[1] = v19;
    v12 = v20;
    v20 = 0;
    *((_DWORD *)a2 + 4) = v12;
    if ( v11 || (v13 = a2[3]) == 0 )
    {
      a2[3] = v21;
      *((_DWORD *)a2 + 8) = v22;
    }
    else
    {
      j_j___libc_free_0_0(v13);
      v11 = v20 <= 0x40;
      a2[3] = v21;
      *((_DWORD *)a2 + 8) = v22;
      if ( !v11 && v19 )
        j_j___libc_free_0_0(v19);
    }
    *a2 |= 4uLL;
  }
  sub_AAF050(&v15, a2 + 1, v4);
  sub_99D930((__int64)&v19, (unsigned __int8 *)(*a2 & 0xFFFFFFFFFFFFFFF8LL), v4, a4[4].m128i_u8[0], 0, 0, 0, 0);
  sub_AB2160(a1, &v15, &v19, 1 - ((unsigned int)(a3 == 0) - 1));
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
