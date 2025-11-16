// Function: sub_82C1B0
// Address: 0x82c1b0
//
__int64 __fastcall sub_82C1B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  char v5; // dl
  __int64 v6; // rsi

  *(_QWORD *)a4 = 0;
  result = a1;
  *(_QWORD *)(a4 + 8) = 0;
  *(_BYTE *)(a4 + 16) = 0;
  *(_QWORD *)(a4 + 24) = a2;
  if ( !a1 )
    goto LABEL_5;
  v5 = *(_BYTE *)(a1 + 80);
  v6 = a1;
  if ( v5 != 16 )
  {
    if ( v5 != 24 )
      goto LABEL_4;
LABEL_7:
    v6 = *(_QWORD *)(v6 + 88);
    if ( *(_BYTE *)(v6 + 80) != 17 )
      goto LABEL_5;
    goto LABEL_6;
  }
  v6 = **(_QWORD **)(a1 + 88);
  v5 = *(_BYTE *)(v6 + 80);
  if ( v5 == 24 )
    goto LABEL_7;
LABEL_4:
  if ( v5 != 17 )
  {
LABEL_5:
    *(_QWORD *)a4 = a1;
    return result;
  }
LABEL_6:
  *(_BYTE *)(a4 + 16) = 1;
  result = *(_QWORD *)(v6 + 88);
  *(_QWORD *)a4 = result;
  return result;
}
