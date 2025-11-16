// Function: sub_73D850
// Address: 0x73d850
//
__int64 __fastcall sub_73D850(__int64 a1)
{
  __int64 v1; // rax
  char v2; // bl
  __int64 *v3; // r13
  __m128i *v4; // rax
  __int64 v5; // r13
  __m128i *v6; // r12

  v1 = *(_QWORD *)(a1 + 72);
  v2 = *(_BYTE *)(a1 + 56);
  v3 = *(__int64 **)(v1 + 16);
  v4 = sub_73D720(*(const __m128i **)v1);
  v5 = *v3;
  v6 = v4;
  if ( (unsigned __int8)(v2 - 79) <= 1u )
  {
    if ( dword_4F077C4 == 1 )
      return v5;
    return sub_8D6540(v4);
  }
  else
  {
    if ( (unsigned __int8)(v2 - 84) > 1u )
      return v5;
    if ( !(unsigned int)sub_8D29A0(v4) )
      return (__int64)v6;
    return v5;
  }
}
