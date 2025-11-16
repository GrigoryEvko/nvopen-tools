// Function: sub_8B17A0
// Address: 0x8b17a0
//
void sub_8B17A0()
{
  __int64 v0; // rbx
  char v1; // al
  __int64 v2; // r12
  char v3; // al

  v0 = qword_4F601F0;
  if ( qword_4F601F0 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v0 + 80) & 2) != 0 )
        goto LABEL_8;
      v2 = *(_QWORD *)(v0 + 16);
      v3 = *(_BYTE *)(v2 + 28);
      if ( (v3 & 8) == 0 )
      {
        sub_891B10(v0);
        v3 = *(_BYTE *)(v2 + 28);
      }
      if ( (v3 & 1) != 0 )
        goto LABEL_8;
      v1 = *(_BYTE *)(v0 + 80);
      if ( v1 < 0 )
        break;
      sub_899CC0(v0, 1, 0);
      if ( dword_4D04734 != 1 )
      {
        v1 = *(_BYTE *)(v0 + 80);
        goto LABEL_6;
      }
LABEL_7:
      if ( (*(_BYTE *)(v2 + 28) & 1) == 0 )
      {
LABEL_15:
        if ( (unsigned int)sub_8A9E70(v0, 1) )
          sub_8AB5A0(v0);
      }
LABEL_8:
      v0 = *(_QWORD *)(v0 + 8);
      if ( !v0 )
        return;
    }
    if ( dword_4D04734 == 1 )
      goto LABEL_15;
LABEL_6:
    if ( (v1 & 1) == 0 )
      goto LABEL_8;
    goto LABEL_7;
  }
}
