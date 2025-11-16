// Function: sub_384CE30
// Address: 0x384ce30
//
void __fastcall sub_384CE30(__int64 a1, int a2)
{
  void *v2; // rax
  __int64 v3; // rax
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  unsigned int v6; // r13d
  unsigned int v7; // ebx
  __int64 v8; // rdx
  __int64 v9; // r12

  v2 = sub_16E8CB0();
  v3 = sub_16E8750((__int64)v2, 2 * a2);
  v4 = *(__m128i **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0x1Bu )
  {
    sub_16E7EE0(v3, "Call Graph SCC Pass Manager\n", 0x1Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CB530);
    qmemcpy(&v4[1], "ass Manager\n", 12);
    *v4 = si128;
    *(_QWORD *)(v3 + 24) += 28LL;
  }
  v6 = a2 + 1;
  v7 = 0;
  while ( *(_DWORD *)(a1 + 192) > v7 )
  {
    v8 = v7++;
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v8);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 136LL))(v9, v6);
    sub_160EBB0(a1 + 160, v9, v6);
  }
}
