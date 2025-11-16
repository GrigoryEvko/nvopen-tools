// Function: sub_2699390
// Address: 0x2699390
//
__int64 __fastcall sub_2699390(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  unsigned __int64 v3; // r12
  char v4; // al
  int v5; // eax
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r12
  char v8; // al
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12

  if ( !*(_BYTE *)(a1 + 120) )
    return 1;
  v2 = *(unsigned __int8 **)(a1 + 112);
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v3 = *(_QWORD *)(v3 + 24);
    v4 = *(_BYTE *)v3;
    if ( *(_BYTE *)v3 > 0x1Cu )
    {
LABEL_6:
      v5 = *(unsigned __int8 *)v3;
      if ( !(_BYTE)v5
        || (unsigned __int8)v5 > 0x1Cu
        && (v10 = (unsigned int)(v5 - 34), (unsigned __int8)v10 <= 0x33u)
        && (v11 = 0x8000000000041LL, _bittest64(&v11, v10)) )
      {
        v6 = v3 & 0xFFFFFFFFFFFFFFFCLL | 2;
      }
      else
      {
        v6 = v3 & 0xFFFFFFFFFFFFFFFCLL;
      }
      nullsub_1518();
      sub_256F570(a2, v6, 0, v2, 1u);
      v7 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
        v7 = *(_QWORD *)(v7 + 24);
      v8 = *(_BYTE *)v7;
      if ( *(_BYTE *)v7 > 0x1Cu )
        goto LABEL_11;
      if ( v8 == 22 )
      {
        if ( !sub_B2FC80(*(_QWORD *)(v7 + 24)) )
        {
          v14 = *(_QWORD *)(*(_QWORD *)(v7 + 24) + 80LL);
          if ( !v14 )
            BUG();
          v15 = *(_QWORD *)(v14 + 32);
          v7 = v15 - 24;
          if ( v15 )
            goto LABEL_11;
          goto LABEL_27;
        }
        v8 = *(_BYTE *)v7;
      }
      if ( !v8 && !sub_B2FC80(v7) )
      {
        v16 = *(_QWORD *)(v7 + 80);
        if ( !v16 )
          BUG();
        v17 = *(_QWORD *)(v16 + 32);
        if ( v17 )
        {
          v7 = v17 - 24;
          goto LABEL_11;
        }
      }
LABEL_27:
      v7 = 0;
LABEL_11:
      sub_2570110(a2, v7);
      return 0;
    }
    if ( v4 == 22 )
    {
      if ( !sub_B2FC80(*(_QWORD *)(v3 + 24)) )
      {
        v12 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 80LL);
        if ( !v12 )
          BUG();
        goto LABEL_21;
      }
      v4 = *(_BYTE *)v3;
    }
    if ( v4 || sub_B2FC80(v3) )
      goto LABEL_41;
    v12 = *(_QWORD *)(v3 + 80);
    if ( !v12 )
      BUG();
LABEL_21:
    v13 = *(_QWORD *)(v12 + 32);
    if ( v13 )
    {
      v3 = v13 - 24;
      goto LABEL_6;
    }
LABEL_41:
    BUG();
  }
  return 1;
}
