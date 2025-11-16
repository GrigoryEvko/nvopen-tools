// Function: sub_2E8FE40
// Address: 0x2e8fe40
//
__int64 __fastcall sub_2E8FE40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  const __m128i *v4; // r14
  __int64 result; // rax
  const __m128i *i; // rbx

  v3 = *(_QWORD *)(a3 + 32);
  v4 = (const __m128i *)(v3 + 40LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF));
  result = 5LL * *(unsigned __int16 *)(*(_QWORD *)(a3 + 16) + 2LL);
  for ( i = (const __m128i *)(v3 + 40LL * *(unsigned __int16 *)(*(_QWORD *)(a3 + 16) + 2LL));
        v4 != i;
        i = (const __m128i *)((char *)i + 40) )
  {
    while ( 1 )
    {
      result = i->m128i_u8[0];
      if ( (_BYTE)result )
        break;
      if ( (i->m128i_i8[3] & 0x20) != 0 )
        goto LABEL_4;
LABEL_5:
      i = (const __m128i *)((char *)i + 40);
      if ( v4 == i )
        return result;
    }
    if ( (_BYTE)result == 12 )
    {
LABEL_4:
      result = sub_2E8EAD0(a1, a2, i);
      goto LABEL_5;
    }
  }
  return result;
}
