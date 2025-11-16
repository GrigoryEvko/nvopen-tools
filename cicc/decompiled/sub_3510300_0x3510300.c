// Function: sub_3510300
// Address: 0x3510300
//
bool __fastcall sub_3510300(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx

  v2 = a1;
  if ( a2 != a1 )
  {
    while ( (unsigned int)*(unsigned __int16 *)(v2 + 68) - 1 > 1 || !sub_2E8B090(v2) )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( a2 == v2 )
        return 0;
    }
  }
  return a2 != v2;
}
