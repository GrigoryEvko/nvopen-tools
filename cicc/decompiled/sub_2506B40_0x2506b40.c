// Function: sub_2506B40
// Address: 0x2506b40
//
__int64 __fastcall sub_2506B40(__int64 a1, int *a2, __int64 a3, __int64 a4)
{
  int v6; // esi
  __int64 result; // rax
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v6 = *a2;
  v8[0] = a3;
  result = sub_A73170(v8, v6);
  if ( (_BYTE)result )
    *(_QWORD *)(a4 + 8 * ((unsigned __int64)(unsigned int)*a2 >> 6)) |= 1LL << *a2;
  return result;
}
