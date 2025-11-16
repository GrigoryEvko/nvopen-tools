// Function: sub_127FEC0
// Address: 0x127fec0
//
__int64 __fastcall sub_127FEC0(__int64 a1, __int64 a2)
{
  int v2; // r15d
  unsigned __int8 v3; // bl
  int v4; // eax

  if ( sub_127B420(*(_QWORD *)a2) )
    sub_127B550("cannot evaluate expression with aggregate type as bool!", (_DWORD *)(a2 + 36), 1);
  v2 = sub_1643320(*(_QWORD *)(a1 + 40));
  v3 = sub_127B3A0(*(_QWORD *)a2);
  v4 = sub_128F980(a1, a2);
  return sub_128B420(a1, v4, v3, v2, 0, 0, a2 + 36);
}
