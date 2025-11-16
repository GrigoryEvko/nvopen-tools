// Function: sub_868D90
// Address: 0x868d90
//
__int64 __fastcall sub_868D90(_QWORD *a1, __int64 *a2, __int64 a3, int a4, int a5)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rbx
  _QWORD *v10; // r14
  __int64 result; // rax
  int v12; // r13d
  void *v13; // rsi
  int v14; // r8d
  unsigned int v15; // r9d
  bool v16; // zf
  unsigned int v17; // r9d
  unsigned __int64 v18; // rsi
  int v19; // eax
  _QWORD *v20; // rax
  int *v21; // rdi
  __int64 v22; // rdx
  int v23; // eax
  _QWORD *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  const __m128i *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // r13d
  __int64 v37; // rcx
  __int64 *v38; // rax
  char v39; // r11
  int v40; // edx
  unsigned int v41; // edx
  __int64 v42; // rdx
  int v43; // r10d
  int v44; // r14d
  char v45; // r9
  char v46; // r8
  __int64 v47; // [rsp-10h] [rbp-60h]
  unsigned int *v48; // [rsp-8h] [rbp-58h]
  int v49; // [rsp+0h] [rbp-50h]
  int v50; // [rsp+0h] [rbp-50h]
  const __m128i *v51; // [rsp+0h] [rbp-50h]
  int v52; // [rsp+0h] [rbp-50h]
  int v53[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( !dword_4D04408
    || dword_4F04C64 == -1
    || (v7 = qword_4F04C68[0], v8 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v8 + 7) & 1) == 0) )
  {
LABEL_4:
    v9 = 0;
    v10 = 0;
    result = 1;
    goto LABEL_5;
  }
  v12 = a3;
  v10 = qword_4F04C18;
  if ( a5 )
  {
    if ( qword_4F04C18 )
    {
LABEL_11:
      if ( *((_BYTE *)qword_4F04C18 + 44) )
      {
        v9 = qword_4F04C18[1];
        if ( v9 )
        {
          if ( *(_DWORD *)(v9 + 16) == dword_4F06650[0] )
          {
LABEL_14:
            result = 1;
            goto LABEL_5;
          }
        }
      }
    }
  }
  else if ( qword_4F04C18 )
  {
    if ( *((_BYTE *)qword_4F04C18 + 42) )
    {
      v9 = 0;
      result = 1;
      goto LABEL_5;
    }
    goto LABEL_11;
  }
  v13 = &unk_4F04C48;
  v14 = unk_4F04C48;
  if ( unk_4F04C48 == -1 || (a3 = qword_4F04C68[0] + 776LL * unk_4F04C48) == 0 )
  {
LABEL_53:
    if ( dword_4F04C44 != -1 || (*(_BYTE *)(v8 + 6) & 2) != 0 || (unsigned int)sub_867AA0() )
    {
      v10 = (_QWORD *)sub_85B260(v8, v13, a3);
      *v10 = qword_4F04C18;
      qword_4F04C18 = v10;
      goto LABEL_37;
    }
    goto LABEL_4;
  }
  v13 = (void *)dword_4F06650[0];
  while ( 1 )
  {
    if ( *(_BYTE *)(a3 + 4) == 9 )
    {
      v9 = *(_QWORD *)(a3 + 648);
      if ( v9 )
      {
        while ( 1 )
        {
          v15 = *(_DWORD *)(v9 + 16);
          if ( v15 >= dword_4F06650[0] )
            break;
          v9 = *(_QWORD *)v9;
          if ( !v9 )
            goto LABEL_19;
        }
        v16 = dword_4F06650[0] == v15;
        if ( dword_4F06650[0] < v15 )
        {
          do
          {
            v9 = *(_QWORD *)(v9 + 8);
            if ( !v9 )
              goto LABEL_19;
            v17 = *(_DWORD *)(v9 + 16);
            v16 = dword_4F06650[0] == v17;
          }
          while ( dword_4F06650[0] < v17 );
        }
        if ( v16 )
          break;
      }
    }
LABEL_19:
    a3 = *(int *)(a3 + 552);
    if ( (_DWORD)a3 != -1 )
    {
      a3 = qword_4F04C68[0] + 776 * a3;
      if ( a3 )
        continue;
    }
    goto LABEL_53;
  }
  v18 = (unsigned __int64)&dword_4F04C44;
  *(_QWORD *)(a3 + 648) = v9;
  if ( dword_4F04C44 != -1 || (*(_BYTE *)(v8 + 6) & 2) != 0 )
  {
    if ( *(_BYTE *)(v9 + 62) )
      goto LABEL_42;
    goto LABEL_32;
  }
  v52 = a4;
  v32 = sub_867AA0();
  a4 = v52;
  if ( !v32 )
    goto LABEL_47;
  if ( !*(_BYTE *)(v9 + 62) )
  {
LABEL_32:
    if ( *(_BYTE *)(v9 + 63) )
    {
      v49 = a4;
      v19 = sub_85B1E0(v8, &dword_4F04C44);
      a4 = v49;
      if ( v19 )
        goto LABEL_47;
    }
    goto LABEL_34;
  }
  v14 = unk_4F04C48;
  if ( unk_4F04C48 == -1 )
  {
    if ( dword_4F04C44 != -1
      || (v21 = (int *)qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
    {
      BUG();
    }
    goto LABEL_45;
  }
  v7 = qword_4F04C68[0];
  v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
LABEL_42:
  if ( (*(_BYTE *)(v8 + 6) & 6) != 0 )
  {
    v21 = &dword_4F04C44;
    if ( dword_4F04C44 != -1
      || (v21 = &dword_4F04C64,
          v7 = qword_4F04C68[0],
          (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
    {
      v22 = v7 + 776LL * v14;
      v18 = *(_QWORD *)(v22 + 368);
      if ( *(_BYTE *)(v18 + 80) == 19 )
      {
        v18 = *(_QWORD *)(v18 + 88);
        if ( (*(_BYTE *)(v18 + 265) & 1) != 0 )
        {
          while ( 1 )
          {
            v42 = *(int *)(v22 + 552);
            if ( (_DWORD)v42 == -1 )
              break;
            v22 = v7 + 776 * v42;
            if ( !v22 )
              break;
            if ( *(_BYTE *)(v22 + 4) == 9 )
            {
              if ( (*(_BYTE *)(v22 + 6) & 2) == 0 )
                goto LABEL_48;
              break;
            }
          }
        }
      }
    }
LABEL_45:
    if ( *(_BYTE *)(v9 + 63) )
    {
      v50 = a4;
      v23 = sub_85B1E0(v21, v18);
      a4 = v50;
      if ( v23 )
      {
LABEL_47:
        v7 = qword_4F04C68[0];
        goto LABEL_48;
      }
    }
    v38 = *(__int64 **)(v9 + 24);
    if ( v38 )
    {
      v39 = 0;
      v18 = 0;
      v8 = 0;
      do
      {
        a3 = *((unsigned int *)v38 + 8);
        if ( (_DWORD)a3 )
        {
          if ( (_DWORD)a3 != 1 )
            goto LABEL_34;
          v41 = *((_DWORD *)v38 + 12);
          if ( (_DWORD)v18 )
          {
            v39 = 1;
            if ( (unsigned int)v18 > v41 )
              v18 = v41;
          }
          else
          {
            v18 = v41;
            v39 = 1;
          }
        }
        else
        {
          v40 = *(_DWORD *)(v38[8] + 4);
          if ( (int)v8 < v40 )
            v8 = (unsigned int)v40;
        }
        v38 = (__int64 *)*v38;
      }
      while ( v38 );
      v7 = qword_4F04C68[0];
      a3 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( a3 )
      {
LABEL_92:
        v43 = 0;
        v44 = 0;
        v45 = 0;
        do
        {
          v46 = *(_BYTE *)(a3 + 4);
          if ( v46 == 17 )
          {
            if ( (_DWORD)v18 )
            {
              v18 = (unsigned int)(v18 - 1);
              v45 = 1;
            }
          }
          else if ( v46 == 9 )
          {
            v44 -= (((unsigned __int8)v39 & ((unsigned __int8)v45 ^ 1)) == 0) - 1;
            v45 = 0;
            v43 += (*(_BYTE *)(a3 + 6) & 6) == 0;
          }
          else
          {
            v45 = 0;
          }
          a3 = *(int *)(a3 + 552);
          if ( (_DWORD)a3 == -1 )
            break;
          a3 = v7 + 776 * a3;
        }
        while ( a3 );
      }
      else
      {
        v43 = 0;
        v44 = 0;
      }
      if ( v44 >= (int)v8 )
        v8 = (unsigned int)v44;
      if ( v43 >= (int)v8 )
        goto LABEL_47;
LABEL_34:
      if ( dword_4F04C44 == -1
        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0
        && !(unsigned int)sub_867AA0() )
      {
        v10 = 0;
        goto LABEL_14;
      }
      v10 = (_QWORD *)sub_85B260(v8, v18, a3);
      *v10 = qword_4F04C18;
      qword_4F04C18 = v10;
      sub_85FC80();
      *((_BYTE *)v10 + 45) = 1;
LABEL_37:
      v20 = sub_8663A0();
      v10[1] = v20;
      *((_DWORD *)v20 + 4) = dword_4F06650[0];
      if ( v12 )
        *((_BYTE *)v10 + 44) = 1;
      v9 = v10[1];
      result = 1;
      goto LABEL_5;
    }
    v7 = qword_4F04C68[0];
    a3 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( a3 )
    {
      v8 = 0;
      v39 = 0;
      v18 = 0;
      goto LABEL_92;
    }
  }
LABEL_48:
  v24 = (_QWORD *)v9;
  v25 = 776LL * unk_4F04C48 + v7;
  v26 = **(_QWORD **)(v25 + 408);
  v27 = *(_QWORD *)(v25 + 376);
  v53[0] = 0;
  v28 = sub_8680C0(v9, v26, v27, 0, 0, a4, 0, v53);
  if ( v28 )
  {
    v51 = v28;
    v10 = (_QWORD *)sub_85B260(v9, v48, v47);
    v10[2] = v51;
    v24 = v10;
    v29 = qword_4F04C18;
    v10[1] = v9;
    *((_DWORD *)v10 + 12) = 0;
    *v10 = v29;
    qword_4F04C18 = v10;
    *((_WORD *)v10 + 20) = 0;
    sub_85BF70((__int64)v10);
    sub_7ADFE0(v10, v48, v30);
    v31 = v10[2];
    if ( v31 && !*(_BYTE *)(v31 + 17) )
    {
      v10[3] = unk_4F06640;
      result = 1;
      if ( v12 )
        *((_BYTE *)v10 + 44) = 1;
      goto LABEL_5;
    }
  }
  else
  {
    sub_7ADFE0(v9, v48, v47);
    v10 = 0;
  }
  sub_7AEC00(v24, v48);
  result = 0;
  if ( !v12 )
  {
    v36 = *(_DWORD *)(v9 + 20);
    v37 = dword_4F06650[0];
    if ( v36 != dword_4F06650[0] )
    {
      while ( (_DWORD)v37 )
      {
        if ( word_4F06418[0] != 9 )
        {
          sub_7B8B50((unsigned __int64)dword_4F06650, v48, v33, v37, v34, v35);
          v37 = dword_4F06650[0];
          if ( v36 != dword_4F06650[0] )
            continue;
        }
        result = 0;
        goto LABEL_5;
      }
      if ( !v36 || word_4F06418[0] == 9 )
      {
        result = 0;
      }
      else
      {
        sub_7B8B50((unsigned __int64)dword_4F06650, v48, v33, v37, v34, v35);
        result = 0;
      }
    }
  }
LABEL_5:
  *a1 = v10;
  if ( a2 )
    *a2 = v9;
  return result;
}
