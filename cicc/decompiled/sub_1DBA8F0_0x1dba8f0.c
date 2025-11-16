// Function: sub_1DBA8F0
// Address: 0x1dba8f0
//
void __fastcall sub_1DBA8F0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  _QWORD *v5; // rax
  unsigned __int16 *v6; // rdx
  unsigned __int16 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int16 *v10; // rbx
  char v11; // r13
  unsigned int v12; // eax
  __int16 v13; // ax
  unsigned __int16 *v14; // rdx
  unsigned __int16 v15; // r14
  unsigned __int16 v16; // r13
  __int64 v17; // rdx
  __int16 v18; // ax
  __int16 *v19; // rbx
  __int64 v20; // [rsp+0h] [rbp-40h]
  unsigned __int16 v21; // [rsp+8h] [rbp-38h]
  char v22; // [rsp+Bh] [rbp-35h]

  v4 = a3;
  sub_1DC3BD0(a1[36], a1[29], a1[34], a1[35], a1 + 37);
  v5 = (_QWORD *)a1[31];
  if ( !v5 )
    goto LABEL_34;
  v20 = 4 * v4;
  v6 = (unsigned __int16 *)(4 * v4 + v5[6]);
  v7 = *v6;
  v21 = v6[1];
  if ( *v6 )
  {
    v22 = 0;
    while ( 1 )
    {
      v8 = v7;
      if ( v5[7] + 2LL * *(unsigned int *)(v5[1] + 24LL * v7 + 8) )
      {
        v9 = a1[30];
        v10 = (__int16 *)(v5[7] + 2LL * *(unsigned int *)(v5[1] + 24LL * v7 + 8));
        v11 = 1;
        while ( 1 )
        {
          v12 = v7;
          if ( *(_QWORD *)(*(_QWORD *)(v9 + 272) + 8 * v8) )
          {
            sub_1DC3C10(a1[36], a2, v7);
            v9 = a1[30];
            v12 = v7;
          }
          if ( (*(_QWORD *)(*(_QWORD *)(v9 + 304) + 8LL * (v12 >> 6)) & (1LL << v7)) == 0 )
            v11 = 0;
          v13 = *v10++;
          v7 += v13;
          if ( !v13 )
            break;
          v8 = v7;
        }
        v22 |= v11;
      }
      else
      {
        v22 = 1;
      }
      v7 = v21;
      if ( !v21 )
        break;
      v5 = (_QWORD *)a1[31];
      if ( !v5 )
LABEL_35:
        BUG();
      v21 = 0;
    }
    if ( v22 )
      goto LABEL_17;
    v5 = (_QWORD *)a1[31];
    if ( !v5 )
LABEL_34:
      BUG();
  }
  v14 = (unsigned __int16 *)(v5[6] + v20);
  v15 = *v14;
  v16 = v14[1];
  if ( *v14 )
  {
    while ( 1 )
    {
      v17 = v5[7] + 2LL * *(unsigned int *)(v5[1] + 24LL * v15 + 8);
LABEL_26:
      v19 = (__int16 *)v17;
      while ( v19 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(a1[30] + 272LL) + 8LL * v15) )
          sub_1DC5DD0(a1[36], a2, v15, 0xFFFFFFFFLL, 0);
        v18 = *v19;
        v17 = 0;
        ++v19;
        v15 += v18;
        if ( !v18 )
          goto LABEL_26;
      }
      if ( !v16 )
        break;
      v5 = (_QWORD *)a1[31];
      v15 = v16;
      if ( !v5 )
        goto LABEL_35;
      v16 = 0;
    }
  }
LABEL_17:
  if ( LOBYTE(qword_4FC4440[20]) )
    sub_1DB49A0(a2);
}
