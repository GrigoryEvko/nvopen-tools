// Function: sub_2538FF0
// Address: 0x2538ff0
//
char __fastcall sub_2538FF0(__int64 a1, unsigned __int8 (__fastcall *a2)(__int64, _QWORD), __int64 a3)
{
  char result; // al
  _QWORD *v4; // rbx

  result = *(_BYTE *)(a1 + 97);
  if ( result )
  {
    result = *(_BYTE *)(a1 + 296);
    if ( result )
    {
      v4 = (_QWORD *)(*(_QWORD *)(a1 + 248) + 8LL * *(unsigned int *)(a1 + 256));
      return v4 == sub_2537BA0(*(_QWORD **)(a1 + 248), (__int64)v4, a2, a3);
    }
  }
  return result;
}
