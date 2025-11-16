// Function: sub_8967D0
// Address: 0x8967d0
//
void __fastcall sub_8967D0(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rsi
  _QWORD *i; // rbx
  __int64 v9; // rdx
  __int64 v10; // rdi

  v3 = sub_727DC0();
  v4 = a1[9];
  v3[1] = a2;
  *v3 = v4;
  v5 = (_QWORD *)a1[21];
  for ( a1[9] = v3; v5; v5 = (_QWORD *)*v5 )
  {
    while ( 1 )
    {
      v6 = v5[1];
      v7 = *(_QWORD *)(v6 + 88);
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 80) - 4) <= 1u && v7 && (*(_BYTE *)(v7 + 177) & 0x20) == 0 && a2 != v7 )
        break;
      v5 = (_QWORD *)*v5;
      if ( !v5 )
        goto LABEL_9;
    }
    sub_5ED880(a2, v7, 1, 0);
  }
LABEL_9:
  for ( i = (_QWORD *)a1[12]; i; i = (_QWORD *)*i )
  {
    v9 = i[1];
    switch ( *(_BYTE *)(v9 + 80) )
    {
      case 4:
      case 5:
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
        break;
      case 6:
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v10 = *(_QWORD *)(v9 + 88);
        break;
      default:
        v10 = 0;
        break;
    }
    sub_8967D0(v10, a2);
  }
}
