// Function: sub_7DB8B0
// Address: 0x7db8b0
//
__int64 __fastcall sub_7DB8B0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d

  v1 = 0;
  while ( 1 )
  {
    v2 = v1;
    if ( qword_4D03EE0[v1] == a1 )
      break;
    if ( ++v1 == 11 )
      return 11;
  }
  return v2;
}
