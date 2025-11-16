// Function: sub_854C10
// Address: 0x854c10
//
__int64 __fastcall sub_854C10(const __m128i *a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  _QWORD *v3; // rbx

  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  result = (__int64)sub_853ED0(a1);
  *(_QWORD *)(v1 + 440) = result;
  if ( result )
  {
    v3 = (_QWORD *)result;
    do
    {
      result = sub_853BE0((__int64)v3);
      v3[8] = result;
      v3 = (_QWORD *)*v3;
    }
    while ( v3 );
  }
  return result;
}
