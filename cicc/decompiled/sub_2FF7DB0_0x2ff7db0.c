// Function: sub_2FF7DB0
// Address: 0x2ff7db0
//
_WORD *__fastcall sub_2FF7DB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rsi
  _WORD *result; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  unsigned int v10; // eax

  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 6LL);
  for ( result = (_WORD *)(v3 + 14 * v4); (*result & 0x1FFF) == 0x1FFE; result = (_WORD *)(v3 + v7) )
  {
    v8 = *(_QWORD *)(a1 + 192);
    v9 = *(__int64 (**)())(*(_QWORD *)v8 + 224LL);
    if ( v9 == sub_2FF7B60 )
    {
      v7 = 0;
      v4 = 0;
    }
    else
    {
      v10 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v9)(v8, v4, a2, a1);
      v3 = *(_QWORD *)(a1 + 40);
      v4 = v10;
      v7 = 14LL * v10;
    }
  }
  return result;
}
