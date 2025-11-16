// Function: sub_303F840
// Address: 0x303f840
//
__int64 __fastcall sub_303F840(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  unsigned int v6; // eax

  if ( a2 )
  {
    v6 = sub_303E610(a1, a2, a3, a5);
    if ( (unsigned __int8)a4 < (unsigned __int8)v6 )
      a4 = v6;
  }
  if ( (_BYTE)qword_502AF68 && (unsigned __int8)a4 < 2u )
    return 2;
  return a4;
}
