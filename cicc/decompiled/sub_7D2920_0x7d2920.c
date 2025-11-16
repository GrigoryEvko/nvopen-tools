// Function: sub_7D2920
// Address: 0x7d2920
//
const __m128i *__fastcall sub_7D2920(__int64 a1, __int64 a2)
{
  char v2; // dl
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  const __m128i *result; // rax
  __int64 v7; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_BYTE *)(a2 + 140);
  if ( v2 != 12 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
    if ( dword_4F04C44 != -1 || (v7 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v7 + 6) & 6) != 0) )
    {
LABEL_13:
      if ( v2 == 9 && (*(_BYTE *)(*(_QWORD *)(a2 + 168) + 109LL) & 0x20) != 0 )
        goto LABEL_9;
      goto LABEL_5;
    }
LABEL_12:
    if ( *(_BYTE *)(v7 + 4) != 12 )
      goto LABEL_9;
    goto LABEL_13;
  }
  v3 = a2;
  do
    v3 = *(_QWORD *)(v3 + 160);
  while ( *(_BYTE *)(v3 + 140) == 12 );
  v4 = *(_QWORD *)(*(_QWORD *)v3 + 96LL);
  if ( dword_4F04C44 == -1 )
  {
    v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v7 + 6) & 6) == 0 )
      goto LABEL_12;
  }
LABEL_5:
  v5 = *(_QWORD *)(a1 + 56);
  if ( (*(_BYTE *)(a2 + 177) & 0x20) != 0 )
  {
LABEL_6:
    result = (const __m128i *)sub_7CE9E0(*(_QWORD *)a1, a2, 0, *(_BYTE *)(a1 + 16) & 1, v8);
    *(_QWORD *)(v8[0] + 184LL) = v5;
    return result;
  }
  if ( (unsigned int)sub_8DBE70(*(_QWORD *)(a1 + 56)) )
  {
    v5 = *(_QWORD *)(a1 + 56);
    goto LABEL_6;
  }
LABEL_9:
  result = *(const __m128i **)(v4 + 48);
  if ( result )
    return sub_7D24E0(a1, *(_QWORD **)(v4 + 48), 0, 0);
  return result;
}
