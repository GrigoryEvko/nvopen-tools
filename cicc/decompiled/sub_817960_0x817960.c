// Function: sub_817960
// Address: 0x817960
//
__int64 __fastcall sub_817960(__int64 a1, const __m128i *a2, __int64 a3, __int64 *a4)
{
  _QWORD *v6; // rdi
  __int64 result; // rax

  if ( !a3 || *(_BYTE *)(a3 + 140) == 14 && *(_BYTE *)(a3 + 160) == 2 )
  {
    *a4 += 2;
    sub_8238B0(qword_4F18BE0, "il", 2);
  }
  else
  {
    *a4 += 2;
    sub_8238B0(qword_4F18BE0, "tl", 2);
    sub_80F5E0(a3, 0, a4);
  }
  sub_8178D0(a1, a2, a4);
  v6 = (_QWORD *)qword_4F18BE0;
  ++*a4;
  result = v6[2];
  if ( (unsigned __int64)(result + 1) > v6[1] )
  {
    sub_823810(v6);
    v6 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v6[4] + result) = 69;
  ++v6[2];
  return result;
}
