// Function: sub_35DE500
// Address: 0x35de500
//
char __fastcall sub_35DE500(_DWORD *a1, char *a2)
{
  __int64 v3; // rdi
  char v4; // al
  __int64 v6; // rax
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *((_QWORD *)a2 + 1);
  if ( *(_BYTE *)(v3 + 8) != 12 )
    return 0;
  v4 = *a2;
  if ( *a2 == 22 || v4 == 61 )
    return 1;
  if ( v4 == 85 )
  {
    if ( !(unsigned __int8)sub_A74710((_QWORD *)a2 + 9, 0, 79) )
    {
      v6 = *((_QWORD *)a2 - 4);
      if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *((_QWORD *)a2 + 10) )
        return 0;
      v7[0] = *(_QWORD *)(v6 + 120);
      return sub_A74710(v7, 0, 79);
    }
    return 1;
  }
  if ( v4 != 67 )
    return 0;
  return *a1 == (unsigned int)sub_BCB060(v3);
}
