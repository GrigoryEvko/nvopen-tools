// Function: sub_6FF940
// Address: 0x6ff940
//
void __fastcall sub_6FF940(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 i; // rdx
  char v8; // al

  if ( !a1[16] )
    goto LABEL_8;
  v6 = *(_QWORD *)a1;
  for ( i = *(unsigned __int8 *)(*(_QWORD *)a1 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v6 + 140) )
    v6 = *(_QWORD *)(v6 + 160);
  if ( (_BYTE)i )
  {
    v8 = a1[17];
    if ( v8 == 1 )
    {
      if ( sub_6ED0A0((__int64)a1) )
        sub_6ED030((__int64)a1);
      sub_6FED50((__int64)a1, 0, 0, 1, 0, 0);
    }
    else if ( v8 == 2 )
    {
      sub_6FA340((__int64)a1, a2, i, a4, a5, a6);
      sub_6FED50((__int64)a1, 0, 0, 1, 0, 0);
    }
  }
  else
  {
LABEL_8:
    sub_6E6870((__int64)a1);
  }
}
