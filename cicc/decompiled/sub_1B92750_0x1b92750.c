// Function: sub_1B92750
// Address: 0x1b92750
//
__int64 __fastcall sub_1B92750(__int64 a1, __int64 **a2, unsigned int a3)
{
  int v3; // ebx
  int v4; // ebx

  if ( *((_BYTE *)*a2 + 8) && a3 != 1 )
    sub_16463B0(*a2, a3);
  v3 = sub_14A3620(*(_QWORD *)(a1 + 328));
  v4 = sub_14A34A0(*(_QWORD *)(a1 + 328)) + v3;
  return v4 + (unsigned int)sub_14A3380(*(_QWORD *)(a1 + 328));
}
