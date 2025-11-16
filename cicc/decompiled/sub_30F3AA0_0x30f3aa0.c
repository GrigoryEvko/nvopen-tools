// Function: sub_30F3AA0
// Address: 0x30f3aa0
//
void __fastcall sub_30F3AA0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rbp
  __int64 *v4; // rax
  int v7; // ecx
  bool v8; // sf
  bool v9; // of
  __int64 v10; // rdi
  __int64 *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  __m128i v14; // xmm0
  __int64 *v15; // rdx
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r11
  bool v20; // dl
  int v21; // edx
  __int64 *v22; // rbx
  __int64 v23; // [rsp-10h] [rbp-10h]
  __int64 v24; // [rsp-8h] [rbp-8h]

  if ( (__int64 *)a1 != a2 )
  {
    v4 = (__int64 *)(a1 + 24);
    if ( a2 != (__int64 *)(a1 + 24) )
    {
      v24 = v3;
      v23 = v2;
      while ( 1 )
      {
        v7 = *((_DWORD *)v4 + 4);
        v9 = __OFSUB__(*(_DWORD *)(a1 + 16), v7);
        v8 = *(_DWORD *)(a1 + 16) - v7 < 0;
        if ( *(_DWORD *)(a1 + 16) == v7 )
        {
          v10 = v4[1];
          v9 = __OFSUB__(*(_QWORD *)(a1 + 8), v10);
          v8 = *(_QWORD *)(a1 + 8) - v10 < 0;
        }
        v11 = v4 + 3;
        v12 = *v4;
        if ( v8 != v9 )
        {
          v13 = (__int64)v4 - a1;
          v14 = _mm_loadu_si128((const __m128i *)(v11 - 3));
          *(&v24 - 4) = *(v11 - 1);
          v15 = v11;
          v16 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
          *((__m128i *)&v24 - 3) = v14;
          if ( v13 > 0 )
          {
            do
            {
              v17 = *(v15 - 6);
              v15 -= 3;
              *v15 = v17;
              v15[1] = *(v15 - 2);
              *((_DWORD *)v15 + 4) = *((_DWORD *)v15 - 2);
              --v16;
            }
            while ( v16 );
          }
          v18 = *(&v24 - 5);
          *(_QWORD *)a1 = v12;
          *(_QWORD *)(a1 + 8) = v18;
          *(_DWORD *)(a1 + 16) = *((_DWORD *)&v24 - 8);
          if ( a2 == v11 )
            return;
        }
        else
        {
          v19 = v4[1];
          while ( 1 )
          {
            v21 = *((_DWORD *)v4 - 2);
            v22 = v4;
            v20 = v7 == v21 ? v19 > *(v4 - 2) : v21 < v7;
            v4 -= 3;
            if ( !v20 )
              break;
            v4[3] = *v4;
            v4[4] = v4[1];
            *((_DWORD *)v4 + 10) = *((_DWORD *)v4 + 4);
          }
          *v22 = v12;
          v22[1] = v19;
          *((_DWORD *)v22 + 4) = v7;
          if ( a2 == v11 )
            return;
        }
        v4 = v11;
      }
    }
  }
}
