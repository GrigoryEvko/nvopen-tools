// Function: sub_11C54E0
// Address: 0x11c54e0
//
__int64 __fastcall sub_11C54E0(__int64 a1)
{
  unsigned int v1; // r14d
  int v2; // r13d
  unsigned __int64 v3; // rbx

  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL) == 7 || (unsigned __int8)sub_B2D630(a1, 40) )
  {
    v1 = 0;
  }
  else
  {
    v1 = 1;
    sub_B2D390(a1, 40);
  }
  if ( *(_QWORD *)(a1 + 104) )
  {
    v2 = 0;
    LODWORD(v3) = 0;
    do
    {
      while ( (unsigned __int8)sub_B2D640(a1, v3, 40) )
      {
        v3 = (unsigned int)(v3 + 1);
        if ( v3 >= *(_QWORD *)(a1 + 104) )
          goto LABEL_8;
      }
      v2 = 1;
      sub_B2D3C0(a1, v3, 40);
      v3 = (unsigned int)(v3 + 1);
    }
    while ( v3 < *(_QWORD *)(a1 + 104) );
LABEL_8:
    v1 |= v2;
  }
  return v1;
}
