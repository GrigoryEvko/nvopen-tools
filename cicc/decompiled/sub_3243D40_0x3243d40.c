// Function: sub_3243D40
// Address: 0x3243d40
//
void __fastcall sub_3243D40(__int64 a1)
{
  unsigned int v1; // esi
  unsigned int v2; // edx

  v1 = *(unsigned __int16 *)(a1 + 96);
  if ( (_WORD)v1 )
  {
    v2 = *(unsigned __int16 *)(a1 + 98);
    if ( (_WORD)v2 )
      sub_32422A0((_QWORD *)a1, v1, v2);
  }
}
