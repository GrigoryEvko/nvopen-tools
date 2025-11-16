// Function: sub_7E7A90
// Address: 0x7e7a90
//
void __fastcall sub_7E7A90(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r13
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r14
  int v11; // eax
  _BYTE *v12; // rax

  v3 = a2;
  if ( !a2 )
  {
    v3 = qword_4D03F68[1];
    v8 = qword_4D03F68[10];
    if ( v8 )
    {
      if ( *(_BYTE *)(a1 + 136) > 2u )
      {
        v9 = *(_QWORD *)(v8 + 8);
        v10 = *(_QWORD *)(v9 + 80);
        if ( *(_QWORD *)(v10 + 8) )
        {
          v3 = *(_QWORD *)(v10 + 8);
        }
        else if ( (*(_BYTE *)(v10 + 24) & 4) == 0
               && !*(_QWORD *)(v3 + 160)
               && !*(_QWORD *)(v3 + 232)
               && *(char *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 192LL) >= 0 )
        {
          v11 = sub_880E90();
          v12 = sub_726EB0(2, v11, 0);
          *(_QWORD *)(v10 + 8) = v12;
          *((_QWORD *)v12 + 10) = v9;
          *(_QWORD *)(v3 + 160) = v12;
          v3 = (__int64)v12;
        }
      }
    }
  }
  v5 = *(_BYTE *)(v3 + 28);
  if ( ((v5 - 15) & 0xFD) != 0 && v5 != 2 )
  {
    if ( a3 )
    {
      v6 = qword_4F04C50;
      if ( qword_4F04C50 )
        goto LABEL_5;
    }
LABEL_8:
    sub_72FC40(a1, v3);
    return;
  }
  *(_BYTE *)(a1 + 89) |= 1u;
  v6 = qword_4F04C50;
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(qword_4F04C50 + 32LL);
  if ( !a3 )
    goto LABEL_8;
LABEL_5:
  v7 = *(_QWORD *)(v6 + 32);
  if ( (*(_BYTE *)(v7 + 206) & 1) == 0 && !unk_4D03EB8 || *(_BYTE *)(a1 + 136) > 2u )
    goto LABEL_8;
  sub_7E7700(a1, v3, v7);
}
