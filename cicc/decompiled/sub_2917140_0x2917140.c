// Function: sub_2917140
// Address: 0x2917140
//
void __fastcall sub_2917140(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rbx
  unsigned int v9; // r14d
  _BYTE *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned int v14; // r15d
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // r14
  _BOOL8 v17; // r8
  unsigned int v18; // r14d
  int v19; // eax
  unsigned __int64 v20; // rcx
  unsigned int v21; // eax
  char v22; // dl
  int v23; // eax
  unsigned int v24; // r15d
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v8 = *(_QWORD *)(a2 + 32 * (2 - v7));
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 > 0x40 )
    {
      if ( v9 - (unsigned int)sub_C444A0(v8 + 24) <= 0x40 && !**(_QWORD **)(v8 + 24) )
        goto LABEL_9;
    }
    else if ( !*(_QWORD *)(v8 + 24) )
    {
      goto LABEL_9;
    }
    if ( !*(_BYTE *)(a1 + 344) )
    {
LABEL_5:
      *(_QWORD *)(a1 + 8) = a2;
      return;
    }
  }
  else
  {
    v8 = 0;
    if ( !*(_BYTE *)(a1 + 344) )
      goto LABEL_5;
  }
  v14 = *(_DWORD *)(a1 + 360);
  v7 = a1 + 352;
  if ( v14 <= 0x40 )
  {
    v15 = *(_QWORD *)(a1 + 352);
    goto LABEL_23;
  }
  v21 = sub_C444A0(a1 + 352);
  v7 = a1 + 352;
  a5 = v21;
  if ( v14 - v21 > 0x40 )
  {
LABEL_9:
    v10 = *(_BYTE **)(a1 + 376);
    goto LABEL_10;
  }
  v15 = **(_QWORD **)(a1 + 352);
LABEL_23:
  v10 = *(_BYTE **)(a1 + 376);
  v16 = *(_QWORD *)(a1 + 368);
  a4 = (unsigned __int8)*v10;
  if ( v16 > v15 )
  {
    if ( !(_BYTE)a4 )
    {
      v17 = v8 != 0;
      if ( v8 )
      {
        v18 = *(_DWORD *)(v8 + 32);
        if ( v18 <= 0x40 )
        {
          v20 = *(_QWORD *)(v8 + 24);
        }
        else
        {
          v26 = v7;
          v19 = sub_C444A0(v8 + 24);
          v7 = v26;
          v17 = v8 != 0;
          v20 = -1;
          if ( v18 - v19 <= 0x40 )
            v20 = **(_QWORD **)(v8 + 24);
        }
      }
      else
      {
        if ( v14 <= 0x40 )
        {
          v25 = *(_QWORD *)(a1 + 352);
        }
        else
        {
          v27 = v7;
          v23 = sub_C444A0(v7);
          v7 = v27;
          v17 = 0;
          v24 = v14 - v23;
          v25 = -1;
          if ( v24 <= 0x40 )
            v25 = **(_QWORD **)(a1 + 352);
        }
        v20 = v16 - v25;
      }
      sub_2916EE0(a1, a2, (__int64 *)v7, v20, v17);
    }
    return;
  }
LABEL_10:
  if ( *v10 )
    return;
  if ( !*(_BYTE *)(a1 + 572) )
    goto LABEL_33;
  v11 = *(_QWORD **)(a1 + 552);
  a4 = *(unsigned int *)(a1 + 564);
  v7 = (__int64)&v11[a4];
  if ( v11 == (_QWORD *)v7 )
  {
LABEL_15:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 560) )
    {
      *(_DWORD *)(a1 + 564) = a4 + 1;
      *(_QWORD *)v7 = a2;
      ++*(_QWORD *)(a1 + 544);
LABEL_17:
      v12 = *(_QWORD *)(a1 + 376);
      v13 = *(unsigned int *)(v12 + 240);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 244) )
      {
        sub_C8D5F0(v12 + 232, (const void *)(v12 + 248), v13 + 1, 8u, a5, a6);
        v13 = *(unsigned int *)(v12 + 240);
      }
      *(_QWORD *)(*(_QWORD *)(v12 + 232) + 8 * v13) = a2;
      ++*(_DWORD *)(v12 + 240);
      return;
    }
LABEL_33:
    sub_C8CC70(a1 + 544, a2, v7, a4, a5, a6);
    if ( !v22 )
      return;
    goto LABEL_17;
  }
  while ( a2 != *v11 )
  {
    if ( (_QWORD *)v7 == ++v11 )
      goto LABEL_15;
  }
}
