// Function: sub_1CAD1F0
// Address: 0x1cad1f0
//
bool __fastcall sub_1CAD1F0(__int64 a1)
{
  __int64 v1; // rbx
  bool result; // al
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rdx
  void *v6; // rcx

  v1 = *(_QWORD *)(a1 - 24);
  result = sub_1642F90(*(_QWORD *)v1, 32);
  if ( result )
  {
    v3 = *(unsigned __int8 *)(v1 + 16);
    if ( (unsigned __int8)v3 > 0x17u )
    {
      if ( (unsigned __int8)v3 <= 0x2Fu )
      {
        v4 = 0x80A800000000LL;
        if ( _bittest64(&v4, v3) )
          return (*(_BYTE *)(v1 + 17) & 4) != 0;
      }
      return (_BYTE)v3 == 54;
    }
    if ( (_BYTE)v3 != 5 )
      return (_BYTE)v3 == 54;
    v5 = *(unsigned __int16 *)(v1 + 18);
    result = 0;
    if ( (unsigned __int16)v5 <= 0x17u )
    {
      v6 = &loc_80A800;
      if ( _bittest64((const __int64 *)&v6, v5) )
        return (*(_BYTE *)(v1 + 17) & 4) != 0;
    }
  }
  return result;
}
