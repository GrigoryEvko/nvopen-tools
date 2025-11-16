// Function: sub_737180
// Address: 0x737180
//
__int64 sub_737180()
{
  __int64 result; // rax
  unsigned __int64 v1; // r14
  __int64 v2; // rcx
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // rdx
  char v8; // al
  int v9; // edx
  char v10; // si
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdx

  result = *(_QWORD *)(unk_4F07288 + 104LL);
  if ( !result )
    return result;
  v1 = 0;
  do
  {
    *(_BYTE *)(result + 144) |= 2u;
    result = *(_QWORD *)(result + 112);
    ++v1;
  }
  while ( result );
  qword_4F07A38 = sub_822B10(8 * v1);
  v3 = (__int64 *)qword_4F07A38;
  qword_4F07A30 = qword_4F07A38;
  v4 = unk_4F07288;
  v5 = *(_QWORD *)(unk_4F07288 + 104LL);
  if ( !v5 )
    goto LABEL_16;
  do
  {
    v8 = *(_BYTE *)(v5 + 144);
    if ( (v8 & 4) == 0 )
    {
      v9 = *(unsigned __int8 *)(v5 + 140);
      v10 = v8 & 2;
      v2 = (unsigned int)(v9 - 9);
      if ( (unsigned __int8)(v9 - 9) > 2u && ((_BYTE)v9 != 2 || (*(_BYTE *)(v5 + 161) & 8) == 0) )
      {
        if ( !v10 )
          goto LABEL_10;
        *(_BYTE *)(v5 + 144) |= 4u;
        v6 = 0;
        goto LABEL_9;
      }
      if ( v10 )
      {
        v6 = 1;
        *(_BYTE *)(v5 + 144) = v8 | 0xC;
LABEL_9:
        sub_736EA0(v5, v6);
        v7 = qword_4F07A30 + 8;
        *(_QWORD *)qword_4F07A30 = v5;
        qword_4F07A30 = v7;
      }
    }
LABEL_10:
    v5 = *(_QWORD *)(v5 + 112);
  }
  while ( v5 );
  v3 = (__int64 *)qword_4F07A38;
  v4 = unk_4F07288;
LABEL_16:
  *(_QWORD *)(v4 + 104) = *v3;
  if ( v1 <= 1 )
  {
    v12 = (__int64)v3;
  }
  else
  {
    v11 = v3;
    v12 = (__int64)&v3[v1 - 1];
    do
    {
      v13 = *v11;
      v2 = v11[1];
      ++v11;
      *(_QWORD *)(v13 + 112) = v2;
    }
    while ( (__int64 *)v12 != v11 );
  }
  *(_QWORD *)(*(_QWORD *)v12 + 112LL) = 0;
  v14 = *(_QWORD *)v12;
  qword_4D03FD0[7] = *(_QWORD *)v12;
  result = sub_822B90(v3, 8 * v1, v14, v2);
  qword_4F07A38 = 0;
  return result;
}
