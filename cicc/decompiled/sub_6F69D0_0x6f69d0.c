// Function: sub_6F69D0
// Address: 0x6f69d0
//
void __fastcall sub_6F69D0(_QWORD *a1, unsigned int a2)
{
  unsigned int v2; // r13d
  char v3; // bl
  __int64 v4; // r8
  __int64 v5; // r9
  char v6; // al
  int v7; // esi
  __int64 v8; // r14

  v2 = (a2 >> 4) & 1;
  v3 = a2;
  if ( (unsigned int)sub_8D3410(*a1) )
  {
    if ( (a2 & 2) == 0 )
    {
      sub_6FB570(a1);
      v6 = *((_BYTE *)a1 + 16);
      if ( v6 != 3 )
        goto LABEL_4;
      goto LABEL_12;
    }
    goto LABEL_3;
  }
  if ( (unsigned int)sub_8D2B80(*a1) && (a2 & 0x100) != 0 )
  {
    if ( *((_BYTE *)a1 + 17) != 1 )
      goto LABEL_3;
    if ( !sub_6ED0A0((__int64)a1) && *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
    {
      sub_6FB600(a1);
      goto LABEL_3;
    }
  }
  if ( *((_BYTE *)a1 + 17) != 1 )
  {
LABEL_3:
    v6 = *((_BYTE *)a1 + 16);
    if ( v6 != 3 )
      goto LABEL_4;
LABEL_12:
    if ( (a2 & 8) != 0 )
    {
      if ( *((_BYTE *)a1 + 17) != 3 )
        return;
LABEL_8:
      v7 = v3 & 1;
      goto LABEL_9;
    }
    goto LABEL_32;
  }
  if ( (a2 & 4) == 0 )
  {
    if ( (a2 & 0x80u) != 0 && dword_4F077C4 == 2 && (unsigned int)sub_8D3A70(*a1) && !(unsigned int)sub_8D23B0(*a1) )
      sub_844770(a1, 0);
    else
      sub_6FA3A0(a1);
    goto LABEL_3;
  }
  v6 = *((_BYTE *)a1 + 16);
  if ( v6 != 3 )
  {
LABEL_5:
    if ( v6 == 5 )
    {
      sub_6E65B0(a1[18]);
      sub_6E6840((__int64)a1);
    }
    return;
  }
  if ( (a2 & 8) != 0 )
    return;
LABEL_32:
  sub_6F6890((__m128i *)a1, v2);
  v6 = *((_BYTE *)a1 + 16);
  if ( v6 == 3 )
  {
    v8 = a1[17];
    if ( (unsigned int)sub_6E5430() )
      sub_6854C0(0x12Bu, (FILE *)((char *)a1 + 68), v8);
    sub_6E6840((__int64)a1);
    v6 = *((_BYTE *)a1 + 16);
  }
LABEL_4:
  if ( *((_BYTE *)a1 + 17) != 3 )
    goto LABEL_5;
  v7 = a2 & 0x20;
  if ( v6 != 4 )
    goto LABEL_8;
LABEL_9:
  if ( !v7 )
    sub_6F5FA0((const __m128i *)a1, 0, v2, v2, v4, v5);
}
