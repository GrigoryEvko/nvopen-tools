// Function: sub_A57320
// Address: 0xa57320
//
void __fastcall sub_A57320(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // rbx
  const __m128i *v11; // rcx
  unsigned int v12; // eax
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __m128i *v15; // rsi
  const __m128i *v16; // [rsp+8h] [rbp-48h]
  __m128i v17; // [rsp+10h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  if ( v5 )
  {
    if ( *(_DWORD *)(v5 + 200) )
    {
      v6 = *(_QWORD **)(v5 + 192);
      v9 = &v6[2 * *(unsigned int *)(v5 + 208)];
      if ( v6 != v9 )
      {
        while ( 1 )
        {
          v10 = v6;
          if ( *v6 != -8192 && *v6 != -4096 )
            break;
          v6 += 2;
          if ( v9 == v6 )
            return;
        }
        if ( v6 != v9 )
        {
          v11 = &v17;
          do
          {
            v12 = *((_DWORD *)v10 + 2);
            if ( v12 < a4 && v12 >= a3 )
            {
              v14 = *v10;
              v15 = *(__m128i **)(a2 + 8);
              v17.m128i_i32[0] = *((_DWORD *)v10 + 2);
              v17.m128i_i64[1] = v14;
              if ( v15 == *(__m128i **)(a2 + 16) )
              {
                v16 = v11;
                sub_A571A0((const __m128i **)a2, v15, v11);
                v11 = v16;
              }
              else
              {
                if ( v15 )
                {
                  *v15 = _mm_loadu_si128(&v17);
                  v15 = *(__m128i **)(a2 + 8);
                }
                *(_QWORD *)(a2 + 8) = v15 + 1;
              }
            }
            v13 = v10 + 2;
            if ( v9 == v10 + 2 )
              break;
            while ( 1 )
            {
              v10 = v13;
              if ( *v13 != -8192 && *v13 != -4096 )
                break;
              v13 += 2;
              if ( v9 == v13 )
                return;
            }
          }
          while ( v9 != v13 );
        }
      }
    }
  }
}
