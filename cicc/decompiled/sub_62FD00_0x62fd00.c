// Function: sub_62FD00
// Address: 0x62fd00
//
__int64 __fastcall sub_62FD00(__int64 a1, char a2, int a3, int a4)
{
  char v5; // r14
  __int64 v7; // rax
  char v8; // si
  __int64 v9; // r12
  __int64 v10; // r10
  __int64 i; // rax
  _QWORD *v12; // r11
  _BYTE *v13; // rdx
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-48h]
  _QWORD *v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v5 = a2 & 1;
  v7 = sub_725A70(5);
  v8 = *(_BYTE *)(v7 + 72);
  *(_QWORD *)(v7 + 56) = a1;
  v9 = v7;
  *(_BYTE *)(v7 + 72) = v5 | v8 & 0xFE;
  if ( a1 )
  {
    v10 = a1;
    if ( a3 )
    {
      *(_BYTE *)(a1 + 193) |= 0x40u;
      v10 = *(_QWORD *)(v7 + 56);
    }
    for ( i = *(_QWORD *)(v10 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v12 = **(_QWORD ***)(i + 168);
    if ( (*(_BYTE *)(v9 + 72) & 1) != 0 )
      v12 = (_QWORD *)*v12;
    if ( v12 )
    {
      v20 = unk_4F06BC0;
      if ( unk_4D044B4 )
        goto LABEL_12;
      v13 = (_BYTE *)unk_4F06BC0;
      if ( unk_4F07270 == unk_4F073B8 && (*(_BYTE *)(unk_4F06BC0 - 8LL) & 1) == 0 )
      {
        v13 = *(_BYTE **)(unk_4F07288 + 88LL);
        unk_4F06BC0 = v13;
      }
      if ( *v13 == 4 )
      {
LABEL_12:
        *(_QWORD *)(v9 + 64) = sub_73F570(v10, v12, 0, 1, 1);
      }
      else
      {
        v18 = v10;
        v19 = v12;
        sub_733780(0, 0, 0, 4, 0);
        v15 = unk_4F06BC0;
        *(_QWORD *)(v9 + 64) = sub_73F570(v18, v19, 0, 1, 1);
        if ( v15 )
        {
          if ( !(unsigned int)sub_733920(v15) )
            sub_732E60(v15, 30, v9);
          sub_733F40(0);
        }
      }
      unk_4F06BC0 = v20;
    }
    if ( (*(_BYTE *)(a1 + 193) & 4) != 0 && (a3 & (a4 ^ 1)) != 0 )
    {
      v16 = sub_724D50(0);
      if ( (unsigned int)sub_71AAF0(v9, 1, 1, 1, dword_4F07508, v16) )
      {
        v17 = sub_725A70(2);
        *(_QWORD *)(v17 + 56) = v16;
        v9 = v17;
        if ( (*(_BYTE *)(v16 + 170) & 0x40) != 0 )
          *(_BYTE *)(v17 + 50) |= 0x80u;
      }
    }
  }
  return v9;
}
