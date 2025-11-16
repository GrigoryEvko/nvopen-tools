// Function: sub_7266C0
// Address: 0x7266c0
//
void __fastcall sub_7266C0(__int64 a1, int a2)
{
  __int64 v2; // rax

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 24) &= 0xF00000FF;
  v2 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 28) = v2;
  *(__m128i *)(a1 + 36) = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  sub_7264E0(a1, a2);
}
