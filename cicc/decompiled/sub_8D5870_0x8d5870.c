// Function: sub_8D5870
// Address: 0x8d5870
//
__int64 __fastcall sub_8D5870(__int64 a1)
{
  __int64 v1; // r12
  __int64 **v2; // rbx
  __int64 v3; // rdi
  char i; // al

  v1 = *(_QWORD *)(a1 + 160);
  v2 = **(__int64 ****)(a1 + 168);
  if ( v1 )
  {
    while ( (*(_BYTE *)(v1 + 144) & 0x20) == 0 )
    {
      v3 = sub_8D4130(*(_QWORD *)(v1 + 120));
      for ( i = *(_BYTE *)(v3 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
        v3 = *(_QWORD *)(v3 + 160);
      if ( (unsigned __int8)(i - 9) <= 2u && (unsigned int)sub_8D5870(v3) )
        break;
      v1 = *(_QWORD *)(v1 + 112);
      if ( !v1 )
        goto LABEL_11;
    }
    return 1;
  }
  else
  {
LABEL_11:
    while ( v2 )
    {
      if ( ((_BYTE)v2[12] & 1) != 0 && (unsigned int)sub_8D5870(v2[5]) )
        return 1;
      v2 = (__int64 **)*v2;
    }
    return 0;
  }
}
