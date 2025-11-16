// Function: sub_2631600
// Address: 0x2631600
//
__int64 __fastcall sub_2631600(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  unsigned __int64 *v8; // rdx
  __m128i *v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v15; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]
  __int64 v18[7]; // [rsp+28h] [rbp-38h] BYREF

  LODWORD(v3) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = (__int64)(*(_QWORD *)(a2 + 8) - *(_QWORD *)a2) >> 5;
  if ( (_DWORD)v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 1;
    v6 = v4 + 2;
    v7 = 0;
    v17 = v6;
    do
    {
      while ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
                 a1,
                 (unsigned int)v7,
                 v18) )
      {
        ++v7;
        if ( ++v5 == v17 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v8 = *(unsigned __int64 **)(a2 + 8);
      v9 = *(__m128i **)a2;
      v10 = ((__int64)v8 - *(_QWORD *)a2) >> 5;
      if ( v10 <= v7 )
      {
        if ( v5 > v10 )
        {
          sub_1885B60((__m128i **)a2, v5 - v10);
          v9 = *(__m128i **)a2;
        }
        else if ( v5 < v10 )
        {
          v13 = (unsigned __int64 *)&v9[2 * v5];
          v14 = v13;
          if ( v8 != v13 )
          {
            do
            {
              if ( (unsigned __int64 *)*v13 != v13 + 2 )
              {
                v15 = v8;
                v16 = v13;
                j_j___libc_free_0(*v13);
                v8 = v15;
                v13 = v16;
              }
              v13 += 4;
            }
            while ( v8 != v13 );
            v9 = *(__m128i **)a2;
            *(_QWORD *)(a2 + 8) = v14;
          }
        }
      }
      v11 = v7++;
      ++v5;
      sub_2625F10(a1, (__m128i *)v9[2 * v11].m128i_i64);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v18[0]);
    }
    while ( v5 != v17 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
