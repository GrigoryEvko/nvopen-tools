// Function: sub_ADEAE0
// Address: 0xadeae0
//
void __fastcall sub_ADEAE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *a2;
  v7[0] = v6;
  if ( v6 )
  {
    sub_B96E90(v7, v6, 1);
    v6 = v7[0];
    if ( !a3 )
      goto LABEL_4;
    goto LABEL_3;
  }
  if ( a3 )
  {
LABEL_3:
    sub_BA6610(v6, 4, a3);
    v6 = v7[0];
LABEL_4:
    if ( !a4 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( !a4 )
    goto LABEL_8;
LABEL_5:
  sub_BA6610(v6, 6, a4);
  v6 = v7[0];
LABEL_6:
  *a2 = v6;
  if ( v6 )
  {
    sub_B91220(v7);
    v6 = *a2;
  }
LABEL_8:
  if ( (*(_BYTE *)(v6 + 1) & 0x7F) != 2 && !*(_DWORD *)(v6 - 8) )
  {
    if ( a3 )
      sub_ADDDC0(a1, a3);
    if ( a4 )
      sub_ADDDC0(a1, a4);
  }
}
