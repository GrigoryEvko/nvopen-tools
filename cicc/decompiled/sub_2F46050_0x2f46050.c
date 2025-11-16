// Function: sub_2F46050
// Address: 0x2f46050
//
bool __fastcall sub_2F46050(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  unsigned int v10; // esi
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  unsigned __int64 v15; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  int v23; // r9d
  int v24; // eax
  int v25; // r9d
  int v26; // eax
  int v27; // r9d
  unsigned __int64 v28; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v29[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !*(_BYTE *)a1 )
  {
    sub_2F45270(a1, *(_QWORD *)(a2 + 24));
    v10 = *(_DWORD *)(a1 + 40);
    *(_BYTE *)a1 = 1;
    v11 = *(_QWORD *)(a1 + 24);
    if ( v10 )
    {
      v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( a2 == *v13 )
      {
LABEL_9:
        v28 = v13[1];
        goto LABEL_10;
      }
      v22 = 1;
      while ( v14 != -4096 )
      {
        v23 = v22 + 1;
        v12 = (v10 - 1) & (v22 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_9;
        v22 = v23;
      }
    }
    v13 = (__int64 *)(v11 + 16LL * v10);
    goto LABEL_9;
  }
  sub_2F45710(a1, a2, &v28);
  if ( !*(_BYTE *)a1 )
  {
    sub_2F45270(a1, *(_QWORD *)(a3 + 24));
    v5 = *(_DWORD *)(a1 + 40);
    *(_BYTE *)a1 = 1;
    v6 = *(_QWORD *)(a1 + 24);
    if ( v5 )
    {
      v7 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a3 == *v8 )
      {
LABEL_5:
        v29[0] = v8[1];
        goto LABEL_6;
      }
      v24 = 1;
      while ( v9 != -4096 )
      {
        v25 = v24 + 1;
        v7 = (v5 - 1) & (v24 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a3 == *v8 )
          goto LABEL_5;
        v24 = v25;
      }
    }
    v8 = (__int64 *)(v6 + 16LL * v5);
    goto LABEL_5;
  }
LABEL_10:
  if ( !(unsigned __int8)sub_2F45710(a1, a3, v29) )
    goto LABEL_11;
  if ( !*(_BYTE *)a1 )
  {
    sub_2F45270(a1, *(_QWORD *)(a2 + 24));
    v17 = *(unsigned int *)(a1 + 40);
    *(_BYTE *)a1 = 1;
    v18 = *(_QWORD *)(a1 + 24);
    if ( (_DWORD)v17 )
    {
      v19 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == a2 )
      {
LABEL_18:
        v15 = v20[1];
        return v29[0] > v15;
      }
      v26 = 1;
      while ( v21 != -4096 )
      {
        v27 = v26 + 1;
        v19 = (v17 - 1) & (v26 + v19);
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( a2 == *v20 )
          goto LABEL_18;
        v26 = v27;
      }
    }
    v20 = (__int64 *)(v18 + 16 * v17);
    goto LABEL_18;
  }
LABEL_6:
  sub_2F45710(a1, a2, &v28);
LABEL_11:
  v15 = v28;
  return v29[0] > v15;
}
