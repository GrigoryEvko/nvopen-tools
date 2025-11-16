// Function: sub_7F8700
// Address: 0x7f8700
//
__int64 __fastcall sub_7F8700(__int64 a1)
{
  __int64 v1; // rax
  __m128i *v2; // r12
  _QWORD *v4; // rax

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  if ( HIDWORD(qword_4F0688C) )
  {
    v1 = *(_QWORD *)(a1 + 168);
    if ( (*(_BYTE *)(v1 + 17) & 1) != 0 )
      return sub_72D2E0(*(_QWORD **)(v1 + 40));
  }
  if ( (_DWORD)qword_4F0688C && (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 17LL) & 2) != 0 )
    return sub_7E1C10();
  v2 = sub_73D7F0(a1);
  if ( !(unsigned int)sub_8D2FB0(v2) )
    return (__int64)v2;
  v4 = (_QWORD *)sub_8D46C0(v2);
  return sub_72D2E0(v4);
}
