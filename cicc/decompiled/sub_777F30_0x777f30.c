// Function: sub_777F30
// Address: 0x777f30
//
__int64 __fastcall sub_777F30(__int64 a1, __int64 a2, FILE *a3)
{
  __int64 **v6; // rdi
  __int64 v7; // r8
  _QWORD *v8; // rbx
  char v9; // al
  __int64 **v10; // r12
  __int64 *v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  char i; // dl
  char v16; // dl
  __int64 *v17; // rax
  __int64 v19; // rdx
  char v20; // dl

  v6 = *(__int64 ***)a2;
  v7 = *(_QWORD *)(a2 + 24);
  v8 = **(_QWORD ***)(a2 + 16);
  v9 = *(_BYTE *)(v7 + -((((unsigned int)*(_QWORD *)a2 - (unsigned int)v7) >> 3) + 10))
     & (1 << ((*(_QWORD *)a2 - v7) & 7));
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 202001 )
  {
    if ( v9 || (*(_BYTE *)(v7 - 9) & 1) != 0 )
    {
      v10 = (__int64 **)v8[3];
      v11 = (__int64 *)v8[2];
      v16 = 1;
      goto LABEL_14;
    }
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xABFu, a3, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  v10 = (__int64 **)v8[3];
  v11 = (__int64 *)v8[2];
  if ( !v9 )
    goto LABEL_7;
  v16 = 0;
LABEL_14:
  while ( 1 )
  {
    v17 = *v10;
    if ( *v10 != v11 )
      break;
    v8 = (_QWORD *)*v8;
    if ( !v8 )
      return 1;
    v10 = (__int64 **)v8[3];
    v11 = (__int64 *)v8[2];
  }
  if ( v16 )
  {
    v20 = *(_BYTE *)(a1 + 132) & 0x20;
    if ( v17 )
    {
      if ( !v20 )
      {
        sub_685F20(0xA86u, a3, *v11, *v17, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return 0;
      }
    }
    else if ( !v20 )
    {
      sub_686E10(0xAC1u, a3, *v11, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  if ( v17 )
  {
    if ( (unsigned int)sub_777E50(a1, (int)v10, *(_QWORD *)(v17[5] + 32), *(_QWORD *)(a2 + 24)) )
    {
      *(_BYTE *)(*(_QWORD *)(a2 + 24) - 9LL) &= ~1u;
      v6 = *(__int64 ***)a2;
      goto LABEL_7;
    }
    return 0;
  }
  while ( 1 )
  {
LABEL_7:
    if ( v6 == v10 + 1 )
    {
      v14 = v11[15];
      for ( i = *(_BYTE *)(v14 + 140); i == 12; i = *(_BYTE *)(v14 + 140) )
        v14 = *(_QWORD *)(v14 + 160);
      if ( (unsigned __int8)(i - 9) <= 2u )
        *v6 = 0;
    }
    v12 = *(_QWORD *)(a2 + 24);
    v13 = -(((unsigned int)((_DWORD)v10 - v12) >> 3) + 10);
    *(_BYTE *)(v12 + v13) |= 1 << (((_BYTE)v10 - v12) & 7);
    *v10 = v11;
    v8 = (_QWORD *)*v8;
    if ( !v8 )
      break;
    v10 = (__int64 **)v8[3];
    v11 = (__int64 *)v8[2];
    v6 = *(__int64 ***)a2;
  }
  v19 = *(_QWORD *)(a2 + 24);
  *(_BYTE *)(v19 + -((((unsigned int)*(_QWORD *)a2 - (unsigned int)v19) >> 3) + 10)) |= 1 << ((*(_QWORD *)a2 - v19) & 7);
  *(_BYTE *)(*(_QWORD *)(a2 + 24) - 9LL) &= ~1u;
  return 1;
}
