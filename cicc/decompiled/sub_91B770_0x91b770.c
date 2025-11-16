// Function: sub_91B770
// Address: 0x91b770
//
bool __fastcall sub_91B770(__int64 a1)
{
  int v1; // eax

  while ( 1 )
  {
    v1 = *(unsigned __int8 *)(a1 + 140);
    if ( (_BYTE)v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return (unsigned int)(v1 - 8) <= 3;
}
