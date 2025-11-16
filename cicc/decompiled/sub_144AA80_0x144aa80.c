// Function: sub_144AA80
// Address: 0x144aa80
//
__int64 __fastcall sub_144AA80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  __int64 result; // rax
  unsigned int v8; // r13d
  unsigned int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // r12

  v3 = sub_16E8CB0(a1, a2, a3);
  v4 = sub_16E8750(v3, (unsigned int)(2 * a2));
  v5 = *(__m128i **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0x13u )
  {
    sub_16E7EE0(v4, "Region Pass Manager\n", 20);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428CC40);
    v5[1].m128i_i32[0] = 175269223;
    *v5 = si128;
    *(_QWORD *)(v4 + 24) += 20LL;
  }
  result = *(unsigned int *)(a1 + 192);
  v8 = a2 + 1;
  v9 = 0;
  if ( (_DWORD)result )
  {
    do
    {
      v10 = v9++;
      v11 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v10);
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v11 + 136LL))(v11, v8);
      result = sub_160EBB0(a1 + 160, v11, v8);
    }
    while ( *(_DWORD *)(a1 + 192) > v9 );
  }
  return result;
}
