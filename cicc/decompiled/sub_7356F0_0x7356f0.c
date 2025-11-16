// Function: sub_7356F0
// Address: 0x7356f0
//
void __fastcall sub_7356F0(__int64 a1)
{
  _QWORD *v2; // rdi
  char v3; // al
  __int64 v4; // rax

  if ( !a1 )
  {
    v3 = MEMORY[0x30];
    goto LABEL_10;
  }
  v2 = *(_QWORD **)(a1 + 40);
  if ( v2 )
  {
LABEL_3:
    sub_7347F0(v2);
    goto LABEL_15;
  }
  v3 = *(_BYTE *)(a1 + 48);
  if ( v3 != 3 )
  {
LABEL_10:
    if ( v3 != 6 )
    {
      if ( v3 != 8 )
      {
        if ( (unsigned __int8)(v3 - 3) > 1u )
          goto LABEL_15;
        goto LABEL_8;
      }
      if ( (*(_BYTE *)(a1 + 72) & 1) == 0 )
        goto LABEL_15;
    }
    sub_7357A0(*(_QWORD *)(a1 + 56));
    sub_733B20((_QWORD *)a1);
    return;
  }
  v4 = *(_QWORD *)(a1 + 56);
  if ( *(_BYTE *)(v4 + 24) == 10 )
  {
    v2 = *(_QWORD **)(v4 + 64);
    if ( v2 )
      goto LABEL_3;
  }
LABEL_8:
  sub_735820(*(_QWORD *)(a1 + 56));
LABEL_15:
  sub_733B20((_QWORD *)a1);
}
