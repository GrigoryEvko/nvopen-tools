// Function: sub_88CF10
// Address: 0x88cf10
//
__int64 __fastcall sub_88CF10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  unsigned __int8 v4; // al

  v3 = dword_4F069F8;
  while ( dword_4F069F8 )
  {
    v4 = *(_BYTE *)(a1 + 140);
    if ( v4 == 8 )
    {
      if ( *(char *)(a1 + 142) < 0 )
        return *(unsigned int *)(a1 + 136);
      a1 = sub_8D40F0(a1);
    }
    else
    {
      if ( v4 <= 8u )
      {
        if ( v4 == 2 )
          return *((unsigned int *)&qword_4F60140 + *(unsigned __int8 *)(a1 + 160));
        if ( (unsigned __int8)(v4 - 3) > 2u )
          return *(unsigned int *)(a1 + 136);
        return *((unsigned int *)&qword_4F60100 + *(unsigned __int8 *)(a1 + 160));
      }
      if ( v4 != 12 || *(char *)(a1 + 142) < 0 )
        return *(unsigned int *)(a1 + 136);
      a3 = (unsigned int)qword_4F077B4;
      if ( (_DWORD)qword_4F077B4 && (*(_BYTE *)(a1 + 185) & 8) != 0 )
        return sub_8D4AB0(a1, v3, a3);
      if ( HIDWORD(qword_4F077B4) && (unsigned __int64)(qword_4F077A8 - 30300LL) <= 0x63 )
      {
        do
        {
          a1 = *(_QWORD *)(a1 + 160);
          if ( *(_BYTE *)(a1 + 140) != 12 )
            break;
          a1 = *(_QWORD *)(a1 + 160);
        }
        while ( *(_BYTE *)(a1 + 140) == 12 );
      }
      else
      {
        a1 = *(_QWORD *)(a1 + 160);
      }
    }
  }
  if ( *(char *)(a1 + 142) >= 0 && *(_BYTE *)(a1 + 140) == 12 )
    return sub_8D4AB0(a1, v3, a3);
  else
    return *(unsigned int *)(a1 + 136);
}
