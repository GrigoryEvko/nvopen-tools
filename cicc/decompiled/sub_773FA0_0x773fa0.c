// Function: sub_773FA0
// Address: 0x773fa0
//
__int64 __fastcall sub_773FA0(__int64 a1, __int64 a2, FILE *a3)
{
  __int64 v6; // rdi
  _QWORD **v7; // rsi
  _QWORD **v8; // rax
  __int64 *v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // r10
  unsigned int v12; // r13d
  char v13; // al
  _QWORD *v14; // rax
  __int64 i; // rdx
  __int64 v16; // rax

  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(_QWORD ***)v6;
  v8 = *(_QWORD ***)v6;
  while ( 1 )
  {
    v9 = v8[2];
    v10 = (__int64 *)*v8[3];
    if ( v10 != v9 )
      break;
    v8 = (_QWORD **)*v8;
    if ( !v8 )
    {
      v12 = 1;
      goto LABEL_7;
    }
  }
  v11 = *v9;
  v12 = 0;
  v13 = *(_BYTE *)(a1 + 132) & 0x20;
  if ( v10 )
  {
    if ( !v13 )
    {
      sub_685F20(0xA86u, a3, v11, *v10, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v6 = *(_QWORD *)(a2 + 16);
      v7 = *(_QWORD ***)v6;
    }
  }
  else if ( !v13 )
  {
    sub_686E10(0xAC1u, a3, v11, (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    v6 = *(_QWORD *)(a2 + 16);
    v7 = *(_QWORD ***)v6;
  }
LABEL_7:
  v14 = *v7;
  for ( i = 2; v14; ++i )
  {
    v7 = (_QWORD **)v14;
    v14 = (_QWORD *)*v14;
  }
  *v7 = (_QWORD *)qword_4F08088;
  *(_BYTE *)(a2 + 8) &= ~4u;
  v16 = *(_QWORD *)(v6 + 24);
  qword_4F08080 += i;
  *(_QWORD *)(a2 + 16) = v16;
  qword_4F08088 = v6;
  return v12;
}
