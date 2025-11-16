// Function: sub_F08E70
// Address: 0xf08e70
//
__int64 __fastcall sub_F08E70(unsigned __int8 **a1, unsigned int a2)
{
  __int64 v2; // r12
  unsigned int v4; // eax
  unsigned int v5; // r13d
  __m128i *v6; // rcx
  __int64 v7; // r14
  unsigned int v8; // eax
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int8 *v12; // rdx
  __int64 v13; // r13
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned __int8 *v22; // rbx
  __int64 v23; // rdi
  unsigned int v24; // r14d
  __int64 v25; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+10h] [rbp-30h]
  int v28; // [rsp+18h] [rbp-28h]

  v2 = a2;
  if ( **a1 != (**(_BYTE **)&a1[1][32 * a2 - 64] == 73) )
  {
    v12 = a1[2];
    v13 = **(_QWORD **)v12 + 40LL * a2;
    if ( (*(_QWORD *)v13 & 4) == 0 )
    {
      sub_9AC330((__int64)&v25, *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL, 0, (__m128i *)(*((_QWORD *)v12 + 1) + 96LL));
      if ( *(_DWORD *)(v13 + 16) > 0x40u )
      {
        v19 = *(_QWORD *)(v13 + 8);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
      *(_QWORD *)(v13 + 8) = v25;
      *(_DWORD *)(v13 + 16) = v26;
      v26 = 0;
      if ( *(_DWORD *)(v13 + 32) > 0x40u && (v20 = *(_QWORD *)(v13 + 24)) != 0 )
      {
        j_j___libc_free_0_0(v20);
        v21 = v26;
        *(_QWORD *)(v13 + 24) = v27;
        *(_DWORD *)(v13 + 32) = v28;
        if ( v21 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
      }
      else
      {
        *(_QWORD *)(v13 + 24) = v27;
        *(_DWORD *)(v13 + 32) = v28;
      }
      *(_QWORD *)v13 |= 4uLL;
    }
    v14 = *(_DWORD *)(v13 + 16);
    v15 = *(_QWORD *)(v13 + 8);
    if ( v14 > 0x40 )
      v15 = *(_QWORD *)(v15 + 8LL * ((v14 - 1) >> 6));
    if ( (v15 & (1LL << ((unsigned __int8)v14 - 1))) == 0 )
      return 0;
  }
  v4 = *(_DWORD *)a1[3];
  v5 = *(_DWORD *)a1[4];
  if ( v4 < v5 )
  {
    v6 = (__m128i *)a1[7];
    if ( **a1 )
    {
      *(_DWORD *)&a1[5][4 * v2] = v5
                                - sub_9AF8B0(
                                    *(_QWORD *)&a1[6][8 * v2],
                                    v6[5].m128i_u64[1],
                                    0,
                                    v6[4].m128i_i64[0],
                                    0,
                                    v6[5].m128i_i64[0],
                                    1);
      v4 = *(_DWORD *)a1[3];
    }
    else
    {
      v7 = *(_QWORD *)a1[8] + 40 * v2;
      if ( (*(_QWORD *)v7 & 4) == 0 )
      {
        sub_9AC330((__int64)&v25, *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL, 0, v6 + 6);
        if ( *(_DWORD *)(v7 + 16) > 0x40u )
        {
          v16 = *(_QWORD *)(v7 + 8);
          if ( v16 )
            j_j___libc_free_0_0(v16);
        }
        *(_QWORD *)(v7 + 8) = v25;
        *(_DWORD *)(v7 + 16) = v26;
        v26 = 0;
        if ( *(_DWORD *)(v7 + 32) > 0x40u && (v17 = *(_QWORD *)(v7 + 24)) != 0 )
        {
          j_j___libc_free_0_0(v17);
          v18 = v26;
          *(_QWORD *)(v7 + 24) = v27;
          *(_DWORD *)(v7 + 32) = v28;
          if ( v18 > 0x40 && v25 )
            j_j___libc_free_0_0(v25);
        }
        else
        {
          *(_QWORD *)(v7 + 24) = v27;
          *(_DWORD *)(v7 + 32) = v28;
        }
        *(_QWORD *)v7 |= 4uLL;
      }
      v8 = *(_DWORD *)(v7 + 16);
      if ( v8 > 0x40 )
      {
        v5 -= sub_C44500(v7 + 8);
      }
      else if ( v8 )
      {
        v9 = ~(*(_QWORD *)(v7 + 8) << (64 - (unsigned __int8)v8));
        if ( v9 )
        {
          _BitScanReverse64(&v9, v9);
          v5 -= v9 ^ 0x3F;
        }
        else
        {
          v5 -= 64;
        }
      }
      *(_DWORD *)&a1[5][4 * v2] = v5;
      v4 = *(_DWORD *)a1[3];
    }
  }
  if ( *(_DWORD *)&a1[5][4 * v2] > v4 )
    return 0;
  LODWORD(v10) = **a1;
  if ( !(_BYTE)v10 )
    return 1;
  if ( *a1[1] != 47 )
    return (unsigned int)v10;
  v22 = a1[9];
  v23 = **(_QWORD **)v22 + 40 * v2;
  v10 = (*(__int64 *)v23 >> 2) & 1;
  if ( ((*(__int64 *)v23 >> 2) & 1) != 0 )
  {
    v24 = *(_DWORD *)(v23 + 32);
    if ( v24 <= 0x40 )
    {
      if ( *(_QWORD *)(v23 + 24) )
        return (unsigned int)v10;
    }
    else if ( v24 != (unsigned int)sub_C444A0(v23 + 24) )
    {
      return (unsigned int)v10;
    }
  }
  return sub_9B6260(*(_QWORD *)(*((_QWORD *)v22 + 2) + 8 * v2), (const __m128i *)(*((_QWORD *)v22 + 1) + 96LL), 0);
}
