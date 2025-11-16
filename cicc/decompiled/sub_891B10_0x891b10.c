// Function: sub_891B10
// Address: 0x891b10
//
__int64 __fastcall sub_891B10(__int64 a1)
{
  __int64 v1; // r12
  unsigned int v2; // r8d
  char v3; // r9
  __int64 v4; // r10
  bool v5; // al
  __int64 v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx

  v1 = *(_QWORD *)(a1 + 24);
  v2 = sub_8919F0(a1, 0);
  if ( v2 )
    goto LABEL_6;
  v5 = 0;
  if ( unk_4D03B70 )
    goto LABEL_3;
  if ( ((v3 - 7) & 0xFD) == 0 )
  {
    if ( *(_BYTE *)(v4 + 136) != 2 && !(unsigned int)sub_8D96E0(*(_QWORD *)(v4 + 120)) )
    {
      v3 = *(_BYTE *)(v1 + 80);
      goto LABEL_9;
    }
LABEL_6:
    v5 = 1;
    v2 = 1;
    goto LABEL_3;
  }
LABEL_9:
  if ( (unsigned __int8)(v3 - 10) <= 1u )
  {
    v7 = *(_QWORD *)(v1 + 88);
    if ( *(_BYTE *)(v7 + 172) == 2 || (unsigned int)sub_8D96E0(*(_QWORD *)(v7 + 152)) )
      goto LABEL_6;
  }
  v8 = (_BYTE *)sub_8807C0(v1);
  if ( v8 )
  {
    if ( (v8[124] & 1) != 0 )
      v8 = (_BYTE *)sub_735B70((__int64)v8);
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v8 + 96LL) + 200LL) & 2) != 0 )
      goto LABEL_6;
  }
  v9 = *(_QWORD *)(a1 + 24);
  v10 = *(_QWORD *)(v9 + 88);
  if ( ((*(_BYTE *)(v9 + 80) - 7) & 0xFD) != 0 )
    v5 = (*(_BYTE *)(v10 + 206) & 4) != 0;
  else
    v5 = (*(_BYTE *)(v10 + 175) & 0x10) != 0;
  v2 = v5;
LABEL_3:
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 28LL) = *(_BYTE *)(*(_QWORD *)(a1 + 16) + 28LL) & 0xF7 | (8 * v5);
  return v2;
}
