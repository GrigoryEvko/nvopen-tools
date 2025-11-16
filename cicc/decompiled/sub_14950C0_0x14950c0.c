// Function: sub_14950C0
// Address: 0x14950c0
//
void __fastcall sub_14950C0(__int64 a1, __m128i a2, __m128i a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // rbx
  __int64 v7; // rsi
  __int64 *v8; // rax
  int v9; // edx

  if ( (*(_DWORD *)(a1 + 344))++ == -1 )
  {
    if ( *(_DWORD *)(a1 + 16) )
    {
      v4 = *(_QWORD **)(a1 + 8);
      v5 = &v4[3 * *(unsigned int *)(a1 + 24)];
      if ( v4 != v5 )
      {
        while ( 1 )
        {
          v6 = v4;
          if ( *v4 != -8 && *v4 != -16 )
            break;
          v4 += 3;
          if ( v5 == v4 )
            return;
        }
        if ( v5 != v4 )
        {
          do
          {
            v7 = v6[2];
            v6 += 3;
            v8 = sub_1494E10(*(_QWORD *)(a1 + 112), v7, *(_QWORD *)(a1 + 120), a1 + 128, a2, a3);
            v9 = *(_DWORD *)(a1 + 344);
            *(v6 - 1) = v8;
            *((_DWORD *)v6 - 4) = v9;
            if ( v6 == v5 )
              break;
            while ( *v6 == -16 || *v6 == -8 )
            {
              v6 += 3;
              if ( v5 == v6 )
                return;
            }
          }
          while ( v5 != v6 );
        }
      }
    }
  }
}
