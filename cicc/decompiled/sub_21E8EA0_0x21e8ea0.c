// Function: sub_21E8EA0
// Address: 0x21e8ea0
//
void __fastcall sub_21E8EA0(__int64 a1, unsigned int a2, __int64 a3, const char *a4)
{
  __m128i *v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  _DWORD *v8; // rdx
  int v9; // ebx
  _QWORD *v10; // rdx

  if ( a4 && !strcmp(a4, "mode") )
  {
    v5 = *(__m128i **)(a3 + 24);
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xFu )
    {
      sub_16E7EE0(a3, "barrier.cluster.", 0x10u);
    }
    else
    {
      *v5 = _mm_load_si128((const __m128i *)&xmmword_435F630);
      *(_QWORD *)(a3 + 24) += 16LL;
    }
    if ( (v6 & 0xF) != 0 )
    {
      if ( (v6 & 0xF) != 1 )
        sub_16BD130("bad cluster barrier op", 1u);
      v8 = *(_DWORD **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v8 <= 3u )
      {
        sub_16E7EE0(a3, "wait", 4u);
      }
      else
      {
        *v8 = 1953063287;
        *(_QWORD *)(a3 + 24) += 4LL;
      }
    }
    else
    {
      v7 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v7) <= 5 )
      {
        sub_16E7EE0(a3, "arrive", 6u);
      }
      else
      {
        *(_DWORD *)v7 = 1769108065;
        *(_WORD *)(v7 + 4) = 25974;
        *(_QWORD *)(a3 + 24) += 6LL;
      }
    }
    v9 = (unsigned __int8)v6 >> 4;
    if ( v9 )
    {
      if ( (_BYTE)v9 != 1 )
        sub_16BD130("bad cluster barrier mem mode", 1u);
      v10 = *(_QWORD **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v10 <= 7u )
      {
        sub_16E7EE0(a3, ".relaxed", 8u);
      }
      else
      {
        *v10 = 0x646578616C65722ELL;
        *(_QWORD *)(a3 + 24) += 8LL;
      }
    }
  }
}
