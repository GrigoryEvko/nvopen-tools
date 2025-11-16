// Function: sub_C90690
// Address: 0xc90690
//
_BYTE *__fastcall sub_C90690(__int64 *a1, _QWORD *a2, __int64 a3, unsigned __int8 a4)
{
  __int64 (__fastcall *v5)(__int64, __int64); // rax
  unsigned __int64 v8; // rsi
  int v10; // eax

  v5 = (__int64 (__fastcall *)(__int64, __int64))a1[6];
  if ( v5 )
    return (_BYTE *)v5(a3, a1[7]);
  v8 = *(_QWORD *)(a3 + 8);
  if ( v8 )
  {
    v10 = sub_C8ED90(a1, v8);
    sub_C904A0(a1, *(_QWORD *)(*a1 + 24LL * (unsigned int)(v10 - 1) + 16), (__int64)a2);
  }
  return sub_C8EE80(a3, 0, a2, a4, 1, 1);
}
