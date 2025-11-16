// Function: sub_734850
// Address: 0x734850
//
void __fastcall sub_734850(__int64 a1)
{
  _QWORD *v2; // rdi
  char v3; // al
  __int64 v4; // rax

  if ( !a1 )
  {
    v3 = MEMORY[0x30];
    goto LABEL_4;
  }
  v2 = *(_QWORD **)(a1 + 40);
  if ( v2 )
  {
LABEL_3:
    sub_7347F0(v2);
    v3 = *(_BYTE *)(a1 + 48);
    *(_QWORD *)(a1 + 40) = 0;
    goto LABEL_4;
  }
  v3 = *(_BYTE *)(a1 + 48);
  if ( v3 != 3 )
  {
LABEL_4:
    if ( v3 != 6 )
    {
      if ( v3 != 8 )
      {
        if ( (unsigned __int8)(v3 - 3) > 1u )
        {
LABEL_7:
          sub_733B20((_QWORD *)a1);
          return;
        }
        goto LABEL_13;
      }
      if ( (*(_BYTE *)(a1 + 72) & 1) == 0 )
        goto LABEL_7;
    }
    sub_734910(*(_QWORD *)(a1 + 56));
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
LABEL_13:
  sub_734990(*(_QWORD *)(a1 + 56));
  sub_733B20((_QWORD *)a1);
}
