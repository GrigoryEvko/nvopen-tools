// Function: sub_24915C0
// Address: 0x24915c0
//
__int64 __fastcall sub_24915C0(__int64 a1)
{
  int v1; // eax

  v1 = *(unsigned __int8 *)(a1 + 8);
  switch ( (_BYTE)v1 )
  {
    case 2:
      return 0;
    case 3:
      return 1;
    case 4:
      return 2;
  }
  if ( (unsigned int)(v1 - 17) > 1 )
    BUG();
  return (unsigned int)sub_24915C0(*(_QWORD *)(a1 + 24));
}
