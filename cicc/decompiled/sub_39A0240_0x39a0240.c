// Function: sub_39A0240
// Address: 0x39a0240
//
unsigned __int64 __fastcall sub_39A0240(__int64 a1, unsigned __int8 a2)
{
  _QWORD **v2; // rbx
  unsigned __int64 result; // rax
  _QWORD **i; // r14
  _QWORD *v6; // rsi

  v2 = *(_QWORD ***)(a1 + 168);
  result = *(unsigned int *)(a1 + 176);
  for ( i = &v2[result]; i != v2; result = (unsigned __int64)sub_39A01D0((__int64 *)a1, v6, a2) )
    v6 = *v2++;
  return result;
}
