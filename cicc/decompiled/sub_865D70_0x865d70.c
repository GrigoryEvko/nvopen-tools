// Function: sub_865D70
// Address: 0x865d70
//
__int64 __fastcall sub_865D70(__int64 a1, int a2, unsigned int a3, unsigned int a4, unsigned int a5, int a6)
{
  int v8; // ebx
  int v9; // r13d
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 v14; // al
  char v15; // cl
  char v16; // dl
  __int64 result; // rax
  char v18; // r14
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  unsigned int v21; // [rsp+18h] [rbp-38h]

  v8 = unk_4F04C2C;
  v9 = dword_4F04C64;
  v10 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(a1 + 89) & 1) != 0 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v10 + 80) - 4) > 1u
      || (v13 = *(_QWORD *)(v10 + 88), v14 = *(_BYTE *)(v13 + 177), (v14 & 0x10) == 0) )
    {
      a5 = 0;
      goto LABEL_18;
    }
    a5 = 0;
    v15 = 0;
    if ( a2 )
    {
LABEL_8:
      v15 = v14 >> 7;
      if ( (*(_BYTE *)(v13 + 178) & 1) == 0 )
        v15 = 1;
    }
  }
  else
  {
    if ( (unsigned __int8)(*(_BYTE *)(v10 + 80) - 4) > 1u )
      goto LABEL_3;
    v13 = *(_QWORD *)(v10 + 88);
    v14 = *(_BYTE *)(v13 + 177);
    if ( (v14 & 0x10) == 0 )
      goto LABEL_3;
    v15 = 0;
    if ( a2 )
      goto LABEL_8;
  }
  v16 = *(_BYTE *)(a1 + 178) & 4;
  if ( !unk_4D04238 )
  {
    if ( !v16 )
      goto LABEL_12;
LABEL_13:
    sub_865B60(a1, 0);
LABEL_14:
    sub_85C120(7u, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL) + 24LL), a1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    v12 = dword_4F04C64;
    if ( dword_4F04C64 == -1 )
    {
      MEMORY[8] &= ~0x40u;
      BUG();
    }
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) |= 0xC0u;
    goto LABEL_20;
  }
  if ( v16 )
    goto LABEL_13;
  if ( (v14 & 0x20) == 0 )
  {
    if ( !v15 )
    {
      v18 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
      sub_865B60(a1, 1u);
      if ( v18 == 8 && (*(_BYTE *)(a1 + 89) & 4) == 0 )
        sub_85B070(v9);
      goto LABEL_14;
    }
    goto LABEL_13;
  }
LABEL_12:
  if ( v15 )
    goto LABEL_13;
LABEL_3:
  if ( unk_4F04C48 != -1 && a5 )
  {
    v19 = a4;
    v21 = a3;
    v20 = sub_878CA0();
    sub_860330(v20, a1);
    sub_864700(v20, a1, 0, *(_QWORD *)a1, *(_QWORD *)a1, 0, 0, a6 | 0x84000);
    sub_864420(a1, v21, v19, a5, 1);
    v12 = dword_4F04C64;
    goto LABEL_19;
  }
LABEL_18:
  v9 = sub_864420(a1, a3, a4, a5, 0);
  v12 = dword_4F04C64;
LABEL_19:
  if ( (_DWORD)v12 == -1 )
  {
    MEMORY[0x23C] = 0;
    BUG();
  }
LABEL_20:
  result = qword_4F04C68[0] + 776 * v12;
  *(_DWORD *)(result + 572) = v9;
  *(_DWORD *)(result + 576) = v8;
  return result;
}
