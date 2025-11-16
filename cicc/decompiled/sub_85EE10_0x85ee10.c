// Function: sub_85EE10
// Address: 0x85ee10
//
__int64 __fastcall sub_85EE10(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rbx
  char v8; // al
  int v9; // ebx
  __int64 v10; // rdx
  __int64 *v11; // r13
  __int64 v12; // rdi
  __int64 result; // rax
  unsigned int v14; // edx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rbx
  __int64 v20; // rdx
  _QWORD *v21; // rbx
  __int64 v22; // rdx
  __int64 v25; // [rsp+10h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 24);
  if ( (*(_BYTE *)(v3 + 124) & 1) != 0 )
    v3 = sub_735B70(v3);
  v4 = *(_QWORD *)v3;
  v25 = *(_QWORD *)(*(_QWORD *)v3 + 96LL);
  if ( a2 )
  {
    v5 = a2;
    v6 = qword_4F04C68[0];
    do
    {
      v7 = v5 - v6;
      v8 = *(_BYTE *)(v5 + 4);
      v9 = 1594008481 * (v7 >> 3);
      if ( ((unsigned __int8)(v8 - 3) <= 2u || !v8) && (unsigned int)sub_85ED80(v4, v5) )
        break;
      v10 = *(int *)(v5 + 552);
      if ( (_DWORD)v10 == -1 )
        break;
      v6 = qword_4F04C68[0];
      v5 = qword_4F04C68[0] + 776 * v10;
    }
    while ( v5 );
  }
  else
  {
    v9 = -1;
  }
  v11 = *(__int64 **)(a2 + 536);
  if ( !v11 )
  {
LABEL_21:
    v15 = unk_4D049E8;
    if ( unk_4D049E8 )
      unk_4D049E8 = *unk_4D049E8;
    else
      v15 = sub_823970(40);
    *(_QWORD *)v15 = 0;
    *(_QWORD *)(v15 + 32) = 0xFFFFFFFFLL;
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = a1;
    *(_QWORD *)(v15 + 24) = v25;
    v16 = *(_QWORD *)(a2 + 536);
    *(_DWORD *)(v15 + 32) = v9;
    *(_QWORD *)v15 = v16;
    *(_DWORD *)(v15 + 36) = a3;
    v17 = qword_4F04C68[0];
    *(_QWORD *)(a2 + 536) = v15;
    v18 = v17 + 776LL * v9;
    *(_QWORD *)(v15 + 8) = *(_QWORD *)(v18 + 544);
    *(_QWORD *)(v18 + 544) = v15;
    v19 = *(_QWORD **)(*(_QWORD *)(v3 + 128) + 184LL);
    if ( v19 )
    {
      do
      {
        v20 = a3;
        if ( v19[7] >= (unsigned __int64)a3 )
          v20 = v19[7];
        sub_85EE10(v19, a2, v20);
        v19 = (_QWORD *)*v19;
      }
      while ( v19 );
      v17 = qword_4F04C68[0];
    }
    *(_BYTE *)(v17 + 776LL * dword_4F04C64 + 5) |= 4u;
    result = *(unsigned int *)(v15 + 36);
    goto LABEL_17;
  }
  while ( 1 )
  {
    v12 = *(_QWORD *)(v11[2] + 24);
    if ( (*(_BYTE *)(v12 + 124) & 1) != 0 )
      break;
    if ( v3 == v12 )
      goto LABEL_16;
LABEL_13:
    v11 = (__int64 *)*v11;
    if ( !v11 )
      goto LABEL_21;
  }
  if ( v3 != sub_735B70(v12) )
    goto LABEL_13;
LABEL_16:
  result = *((unsigned int *)v11 + 9);
  if ( (unsigned int)result > a3 )
  {
    *((_DWORD *)v11 + 9) = a3;
    v21 = *(_QWORD **)(*(_QWORD *)(v3 + 128) + 184LL);
    if ( v21 )
    {
      do
      {
        v22 = a3;
        if ( v21[7] >= (unsigned __int64)a3 )
          v22 = v21[7];
        sub_85EE10(v21, a2, v22);
        v21 = (_QWORD *)*v21;
      }
      while ( v21 );
      result = *((unsigned int *)v11 + 9);
    }
    else
    {
      result = a3;
    }
  }
LABEL_17:
  v14 = *(_DWORD *)(v25 + 168);
  if ( !v14 || v14 > (unsigned int)result )
    *(_DWORD *)(v25 + 168) = result;
  return result;
}
