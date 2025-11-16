// Function: sub_684EA0
// Address: 0x684ea0
//
__int64 __fastcall sub_684EA0(__int64 a1)
{
  bool v2; // cf
  bool v3; // zf
  __int64 v4; // rcx
  __int64 v5; // rdi
  unsigned int *v6; // rdx
  unsigned int *v7; // rsi
  char v8; // al
  bool v9; // cf
  bool v10; // zf
  __int64 v11; // rcx
  unsigned int v12; // r13d
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // r13
  _QWORD *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdi
  _QWORD *v27; // rax
  _QWORD *v28; // r9
  __int64 v29; // r8
  _QWORD *v30; // r10
  _QWORD *v31; // rdx
  _QWORD *v32; // rsi
  __int64 v33; // [rsp+0h] [rbp-60h]
  _QWORD *v34; // [rsp+0h] [rbp-60h]
  _QWORD *v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __m128i v37; // [rsp+10h] [rbp-50h] BYREF
  __int64 v38; // [rsp+20h] [rbp-40h]

  sub_7C9660(a1);
  v2 = word_4F06418[0] == 0;
  v3 = word_4F06418[0] == 1;
  if ( word_4F06418[0] != 1 )
  {
    v7 = &dword_4F063F8;
    v5 = 3184;
    v12 = 1;
    sub_684B30(0xC70u, &dword_4F063F8);
    goto LABEL_20;
  }
  v4 = 5;
  v5 = (__int64)"push";
  v6 = *(unsigned int **)(qword_4D04A00 + 8);
  v7 = v6;
  do
  {
    if ( !v4 )
      break;
    v2 = *(_BYTE *)v7 < *(_BYTE *)v5;
    v3 = *(_BYTE *)v7 == *(_BYTE *)v5;
    v7 = (unsigned int *)((char *)v7 + 1);
    ++v5;
    --v4;
  }
  while ( v3 );
  v8 = (!v2 && !v3) - v2;
  v9 = 0;
  v10 = v8 == 0;
  if ( v8 )
  {
    v17 = 4;
    v5 = (__int64)"pop";
    v7 = *(unsigned int **)(qword_4D04A00 + 8);
    do
    {
      if ( !v17 )
        break;
      v9 = *(_BYTE *)v7 < *(_BYTE *)v5;
      v10 = *(_BYTE *)v7 == *(_BYTE *)v5;
      v7 = (unsigned int *)((char *)v7 + 1);
      ++v5;
      --v17;
    }
    while ( v10 );
    v12 = (char)((!v9 && !v10) - v9);
    if ( (!v9 && !v10) != v9 )
    {
      v7 = &dword_4F063F8;
      v5 = 3184;
      v12 = 1;
      sub_684B30(0xC70u, &dword_4F063F8);
      sub_7B8B50(3184, &dword_4F063F8, v19, v20);
      goto LABEL_20;
    }
    sub_7B8B50(v5, v7, v6, v17);
  }
  else
  {
    sub_7B8B50(v5, v7, v6, v4);
    v12 = 1;
  }
  if ( unk_4F04C48 != -1
    && (v13 = (__int64)qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0) )
  {
    if ( (*(_BYTE *)(a1 + 72) & 4) == 0 )
      goto LABEL_24;
    v5 = (__int64)&v37;
    v38 = -1;
    v37.m128i_i16[4] = v37.m128i_i16[4] & 0xFE00 | 0x24;
    v37.m128i_i32[0] = dword_4F063F8;
    v37.m128i_i16[2] = word_4F063FC[0];
    v14 = sub_67F350(&v37);
  }
  else
  {
    v5 = (__int64)&v37;
    v38 = -1;
    v37.m128i_i16[4] = v37.m128i_i16[4] & 0xFE00 | 0x24;
    v37.m128i_i32[0] = dword_4F063F8;
    v37.m128i_i16[2] = word_4F063FC[0];
    sub_67F350(&v37);
    v13 = 3LL * *(_QWORD *)(qword_4CFFD80 + 16) - 3;
    v14 = *(_QWORD *)qword_4CFFD80 + 8 * v13;
  }
  if ( !v14 )
  {
LABEL_24:
    v12 = 0;
    goto LABEL_20;
  }
  v15 = (__int64 *)qword_4CFFD78;
  if ( v12 )
  {
    v21 = *(_QWORD *)(qword_4CFFD78 + 16);
    v22 = *(_QWORD *)(qword_4CFFD78 + 8);
    v23 = *(_QWORD *)(qword_4CFFD80 + 16);
    if ( v21 == v22 )
    {
      if ( v21 <= 1 )
      {
        v26 = 16;
        v25 = 2;
      }
      else
      {
        v25 = v21 + (v21 >> 1) + 1;
        v26 = 8 * v25;
      }
      v33 = v25;
      v35 = *(_QWORD **)qword_4CFFD78;
      v27 = (_QWORD *)sub_823970(v26);
      v28 = v35;
      v29 = v33;
      v30 = v27;
      if ( v21 > 0 )
      {
        v31 = v35;
        v32 = &v27[v21];
        do
        {
          if ( v27 )
            *v27 = *v31;
          ++v27;
          ++v31;
        }
        while ( v27 != v32 );
      }
      v7 = (unsigned int *)(8 * v22);
      v5 = (__int64)v35;
      v34 = v30;
      v36 = v29;
      sub_823A00(v28, 8 * v22);
      *v15 = (__int64)v34;
      v15[1] = v36;
    }
    v24 = (_QWORD *)(*v15 + 8 * v21);
    if ( v24 )
      *v24 = v23 - 1;
    v15[2] = v21 + 1;
    goto LABEL_24;
  }
  *(_BYTE *)(v14 + 9) |= 1u;
  v16 = v15[2];
  if ( v16 )
  {
    v11 = *v15;
    v13 = *(_QWORD *)(*v15 + 8 * v16 - 8);
    *(_QWORD *)(v14 + 16) = v13;
    --v15[2];
  }
  else
  {
    *(_QWORD *)(v14 + 16) = -1;
    v7 = &dword_4F063F8;
    v5 = 3185;
    sub_684B30(0xC71u, &dword_4F063F8);
  }
LABEL_20:
  while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
    sub_7B8B50(v5, v7, v13, v11);
  return sub_7C96B0(v12);
}
