// Function: sub_87EF90
// Address: 0x87ef90
//
_QWORD *__fastcall sub_87EF90(unsigned __int8 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // r12
  __int64 v9; // rdx
  char v10; // al
  __int64 v12; // rdi
  char v13; // al
  __int64 v14; // rax

  v2 = (_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)a2;
  v8 = sub_87EBB0(a1, v4, v2);
  v9 = *(_BYTE *)(a2 + 17) & 0x20;
  v10 = v9 | *((_BYTE *)v8 + 81) & 0xDF;
  *((_BYTE *)v8 + 81) = v10;
  if ( (v10 & 0x20) == 0 )
    goto LABEL_2;
  if ( (*(_BYTE *)(a2 + 18) & 2) != 0 )
  {
    v12 = *(_QWORD *)(a2 + 32);
    v13 = *(_BYTE *)(v12 + 140);
    if ( v13 == 14 )
    {
      v14 = sub_7CFE40(v12, v4, v9, v5, v6, v7);
      v12 = v14;
      if ( !v14 )
      {
LABEL_2:
        *(_QWORD *)(a2 + 24) = v8;
        *(_BYTE *)(a2 + 16) &= ~1u;
        return v8;
      }
      v13 = *(_BYTE *)(v14 + 140);
    }
    if ( (unsigned __int8)(v13 - 9) <= 2u )
    {
      *((_BYTE *)v8 + 81) |= 0x10u;
      v8[8] = v12;
    }
    goto LABEL_2;
  }
  v8[8] = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a2 + 24) = v8;
  *(_BYTE *)(a2 + 16) &= ~1u;
  return v8;
}
