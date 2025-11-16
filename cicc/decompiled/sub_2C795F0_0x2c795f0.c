// Function: sub_2C795F0
// Address: 0x2c795f0
//
void __fastcall sub_2C795F0(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rdi
  __m128i *v10; // rdx
  __m128i si128; // xmm0
  __int64 v12; // rcx
  _BYTE *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdx

  sub_2C79160(a1, *(_QWORD *)(a2 + 8), a2);
  if ( (unsigned __int8)(*(_BYTE *)a2 - 22) > 6u && (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
  {
    v4 = 0;
    v5 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    do
    {
      while ( 1 )
      {
        v6 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v7 = *(_QWORD *)(v6 + v4);
        sub_2C79160(a1, *(_QWORD *)(v7 + 8), a2);
        if ( *(_BYTE *)v7 == 4 )
        {
          if ( *(_BYTE *)a2 <= 0x1Cu )
          {
            if ( *(_BYTE *)a2 == 3 )
            {
              v8 = a2;
              v9 = sub_2C767C0(a1, a2, 0);
            }
            else
            {
              v15 = *(_QWORD *)(a1 + 24);
              v16 = *(_QWORD *)(v15 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 6 )
              {
                v8 = (__int64)"Error: ";
                sub_CB6200(v15, (unsigned __int8 *)"Error: ", 7u);
              }
              else
              {
                v8 = 14962;
                *(_DWORD *)v16 = 1869771333;
                *(_WORD *)(v16 + 4) = 14962;
                *(_BYTE *)(v16 + 6) = 32;
                *(_QWORD *)(v15 + 32) += 7LL;
              }
              v9 = *(_QWORD *)(a1 + 24);
            }
          }
          else
          {
            v8 = a2;
            v9 = sub_2C76A00(a1, a2, 0);
          }
          v10 = *(__m128i **)(v9 + 32);
          if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 0x1Du )
          {
            v8 = (__int64)"blockaddress is not supported\n";
            sub_CB6200(v9, "blockaddress is not supported\n", 0x1Eu);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_42D0970);
            v12 = 2660;
            qmemcpy(&v10[1], "not supported\n", 14);
            *v10 = si128;
            *(_QWORD *)(v9 + 32) += 30LL;
          }
          v13 = *(_BYTE **)(a1 + 16);
          if ( v13 )
            *v13 = 0;
          if ( !*(_DWORD *)(a1 + 4) )
            break;
        }
        v4 += 32;
        if ( v5 == v4 )
          return;
      }
      v14 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v14 + 32) != *(_QWORD *)(v14 + 16) )
      {
        sub_CB5AE0((__int64 *)v14);
        v14 = *(_QWORD *)(a1 + 24);
      }
      v4 += 32;
      sub_CEB520(*(_QWORD **)(v14 + 48), v8, (__int64)v10, (char *)v12);
    }
    while ( v5 != v4 );
  }
}
