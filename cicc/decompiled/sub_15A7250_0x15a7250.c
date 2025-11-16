// Function: sub_15A7250
// Address: 0x15a7250
//
void __fastcall sub_15A7250(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *a2;
  v7[0] = v6;
  if ( v6 )
  {
    sub_1623A60(v7, v6, 2);
    v6 = v7[0];
    if ( !a3 )
      goto LABEL_4;
    goto LABEL_3;
  }
  if ( a3 )
  {
LABEL_3:
    sub_1630830(v6, 4, a3);
    v6 = v7[0];
LABEL_4:
    if ( !a4 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( !a4 )
    goto LABEL_8;
LABEL_5:
  sub_1630830(v6, 6, a4);
  v6 = v7[0];
LABEL_6:
  *a2 = v6;
  if ( v6 )
  {
    sub_161E7C0(v7);
    v6 = *a2;
  }
LABEL_8:
  if ( *(_BYTE *)(v6 + 1) != 2 && !*(_DWORD *)(v6 + 12) )
  {
    if ( a3 )
      sub_15A6B80(a1, a3);
    if ( a4 )
      sub_15A6B80(a1, a4);
  }
}
