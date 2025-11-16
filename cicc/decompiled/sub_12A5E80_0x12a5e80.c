// Function: sub_12A5E80
// Address: 0x12a5e80
//
__int64 __fastcall sub_12A5E80(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // rax
  char v5; // cl

  v3 = *(_BYTE *)(a3 + 140);
  if ( v3 != 12 )
  {
    if ( v3 != 1 )
      goto LABEL_6;
LABEL_11:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0x300000000LL;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v4 = a3;
  do
  {
    v4 = *(_QWORD *)(v4 + 160);
    v5 = *(_BYTE *)(v4 + 140);
  }
  while ( v5 == 12 );
  if ( v5 == 1 )
    goto LABEL_11;
  do
  {
    a3 = *(_QWORD *)(a3 + 160);
    v3 = *(_BYTE *)(a3 + 140);
  }
  while ( v3 == 12 );
LABEL_6:
  if ( v3 == 2 && *(_BYTE *)(a3 + 160) <= 4u )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0x100000000LL;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
}
