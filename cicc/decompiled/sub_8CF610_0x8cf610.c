// Function: sub_8CF610
// Address: 0x8cf610
//
__int64 __fastcall sub_8CF610(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  char v4; // al
  _QWORD *v6; // rax
  __int64 i; // rax

  v2 = *(_QWORD **)(a2 + 32);
  v3 = *(_QWORD **)(a1 + 32);
  *(_QWORD *)(a1 + 32) = v2;
  *v3 = *v2;
  v4 = *(_BYTE *)(a1 + 140);
  if ( v4 != *(_BYTE *)(a2 + 140) )
    return 0;
  if ( (unsigned __int8)(v4 - 9) <= 2u )
  {
    if ( (unsigned int)sub_8D2490(a1) )
    {
      for ( i = *(_QWORD *)(a1 + 160); i; i = *(_QWORD *)(i + 112) )
        *(_QWORD *)(i + 32) = 0;
      sub_8CAE10(a1);
      return sub_8CE860(a1);
    }
    else
    {
      return 1;
    }
  }
  else
  {
    if ( v4 != 2 || (*(_BYTE *)(a1 + 161) & 8) == 0 )
      sub_721090();
    if ( (**(_BYTE **)(a1 + 176) & 1) != 0 )
    {
      v6 = *(_QWORD **)(a1 + 168);
      if ( (*(_BYTE *)(a1 + 161) & 0x10) != 0 )
        v6 = (_QWORD *)v6[12];
      while ( v6 )
      {
        v6[4] = 0;
        v6 = (_QWORD *)v6[15];
      }
    }
    sub_8CA420(a1);
    return sub_8C7CC0(a1);
  }
}
