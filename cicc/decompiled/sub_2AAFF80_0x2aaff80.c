// Function: sub_2AAFF80
// Address: 0x2aaff80
//
__int64 __fastcall sub_2AAFF80(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 result; // rax

  v1 = sub_2BF3F10(a1);
  v2 = sub_2BF04D0(v1);
  if ( (*(_QWORD *)(v2 + 112) & 0xFFFFFFFFFFFFFFF8LL) == v2 + 112 )
  {
    if ( *(_DWORD *)(v2 + 88) != 1 )
      BUG();
    v2 = **(_QWORD **)(v2 + 80);
  }
  result = *(_QWORD *)(v2 + 120);
  if ( result )
    result -= 24;
  return result;
}
