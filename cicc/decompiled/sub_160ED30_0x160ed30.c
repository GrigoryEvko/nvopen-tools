// Function: sub_160ED30
// Address: 0x160ed30
//
void __fastcall sub_160ED30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  unsigned int v7; // r13d
  unsigned int v8; // ebx
  __int64 v9; // rdx
  __int64 v10; // r12

  v3 = sub_16BA580(a1, a2, a3);
  v4 = sub_16E8750(v3, (unsigned int)(2 * a2));
  v5 = *(__m128i **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0x14u )
  {
    sub_16E7EE0(v4, "FunctionPass Manager\n", 21);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F55290);
    v5[1].m128i_i32[0] = 1919248225;
    v5[1].m128i_i8[4] = 10;
    *v5 = si128;
    *(_QWORD *)(v4 + 24) += 21LL;
  }
  v7 = a2 + 1;
  v8 = 0;
  while ( *(_DWORD *)(a1 + 192) > v8 )
  {
    v9 = v8++;
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v9);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v10 + 136LL))(v10, v7);
    sub_160EBB0(a1 + 160, v10, v7);
  }
}
