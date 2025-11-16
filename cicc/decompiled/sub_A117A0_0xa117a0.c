// Function: sub_A117A0
// Address: 0xa117a0
//
__int64 __fastcall sub_A117A0(__int64 a1, unsigned int a2)
{
  __m128i *v3; // rdi
  unsigned __int64 v4; // rax
  unsigned int v5; // edx
  _BYTE *v6; // r8
  __int64 *v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r8
  __int64 v11; // rcx
  unsigned int v12[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *(__m128i **)a1;
  v4 = (v3[45].m128i_i64[0] - v3[44].m128i_i64[1]) >> 4;
  if ( a2 < v4 )
    return sub_A08720((__int64)v3, a2);
  v5 = v3->m128i_u32[2];
  if ( **(_BYTE **)(a1 + 8) )
  {
    if ( a2 >= v5
      || (v6 = *(_BYTE **)(v3->m128i_i64[0] + 8LL * a2)) == 0
      || (unsigned __int8)(*v6 - 5) <= 0x1Fu && ((v6[1] & 0x7F) == 2 || *((_DWORD *)v6 - 2)) )
    {
      v7 = *(__int64 **)(a1 + 24);
      v12[0] = a2;
      v8 = v7[6];
      if ( v8 == v7[8] - 16 )
      {
        sub_A03F10(v7, (int *)v12);
        v9 = v7[6];
      }
      else
      {
        if ( v8 )
        {
          *(_DWORD *)v8 = 259;
          *(_QWORD *)(v8 + 8) = 0;
          *(_DWORD *)(v8 + 4) = a2;
          v8 = v7[6];
        }
        v9 = v8 + 16;
        v7[6] = v9;
      }
      if ( v7[7] == v9 )
        v9 = *(_QWORD *)(v7[9] - 8) + 512LL;
      return v9 - 16;
    }
    return (__int64)v6;
  }
  if ( a2 < v5 )
  {
    v6 = *(_BYTE **)(v3->m128i_i64[0] + 8LL * a2);
    if ( v6 )
      return (__int64)v6;
  }
  if ( a2 < ((v3[46].m128i_i64[1] - v3[46].m128i_i64[0]) >> 3) + v4 )
  {
    sub_A07560((__int64)v3, **(_DWORD **)(a1 + 16));
    sub_A0FFA0(*(__m128i **)a1, a2, *(_QWORD *)(a1 + 24), v11);
    v6 = 0;
    if ( a2 < *(_DWORD *)(*(_QWORD *)a1 + 8LL) )
      return *(_QWORD *)(**(_QWORD **)a1 + 8LL * a2);
    return (__int64)v6;
  }
  return sub_A07560((__int64)v3, a2);
}
