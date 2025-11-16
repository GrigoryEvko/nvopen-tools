// Function: sub_2AB42B0
// Address: 0x2ab42b0
//
bool __fastcall sub_2AB42B0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v5; // eax
  unsigned int v6; // edx

  if ( !(unsigned __int8)sub_DFE3D0(*(_QWORD *)(a1 + 448)) || (unsigned int)sub_DFB730(*(_QWORD *)(a1 + 448)) <= 1 )
    return 0;
  if ( BYTE4(a2) )
    a3 = 1;
  if ( (int)sub_23DF0D0(&dword_500EB48) <= 0 )
    v5 = sub_DF9C90(*(_QWORD *)(a1 + 448));
  else
    v5 = qword_500EBC8;
  v6 = a3 * a2;
  if ( BYTE4(a2) )
  {
    if ( *(_BYTE *)(a1 + 8) )
      v6 *= *(_DWORD *)(a1 + 4);
  }
  return v5 <= v6;
}
