// Function: sub_2DF7410
// Address: 0x2df7410
//
void __fastcall sub_2DF7410(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned __int8 *v8; // rax
  unsigned int v9; // r12d
  size_t v10; // rdx
  size_t v11; // r14
  void *v12; // rdi
  _BYTE *v13; // r14
  __int64 v14; // rdi
  unsigned __int8 v15; // al
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int64 v18; // rdi
  size_t v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // rdx
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 v25[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_BYTE *)a2 == 26 )
  {
    v5 = *(_BYTE *)(a2 - 16);
    if ( (v5 & 2) != 0 )
      v6 = *(_QWORD *)(a2 - 32);
    else
      v6 = a2 - 16 - 8LL * ((v5 >> 2) & 0xF);
    v7 = *(_QWORD *)(v6 + 8);
    if ( !v7 )
      goto LABEL_12;
    v8 = (unsigned __int8 *)sub_B91420(v7);
    v9 = *(_DWORD *)(a2 + 16);
    v11 = v10;
  }
  else
  {
    if ( *(_BYTE *)a2 != 27 )
      goto LABEL_12;
    v16 = *(_BYTE *)(a2 - 16);
    v17 = (v16 & 2) != 0 ? *(_QWORD *)(a2 - 32) : a2 - 16 - 8LL * ((v16 >> 2) & 0xF);
    v18 = *(_QWORD *)(v17 + 8);
    if ( !v18 )
      goto LABEL_12;
    v8 = (unsigned __int8 *)sub_B91420(v18);
    v9 = *(_DWORD *)(a2 + 4);
    v11 = v19;
  }
  if ( v11 )
  {
    v12 = *(void **)(a1 + 32);
    if ( v11 > *(_QWORD *)(a1 + 24) - (_QWORD)v12 )
    {
      v24 = sub_CB6200(a1, v8, v11);
      v13 = *(_BYTE **)(v24 + 32);
      v14 = v24;
    }
    else
    {
      memcpy(v12, v8, v11);
      v13 = (_BYTE *)(*(_QWORD *)(a1 + 32) + v11);
      v14 = a1;
      *(_QWORD *)(a1 + 32) = v13;
    }
    if ( v13 == *(_BYTE **)(v14 + 24) )
    {
      v14 = sub_CB6200(v14, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *v13 = 44;
      ++*(_QWORD *)(v14 + 32);
    }
    sub_CB59D0(v14, v9);
  }
LABEL_12:
  if ( !a3 )
    return;
  v15 = *(_BYTE *)(a3 - 16);
  if ( (v15 & 2) != 0 )
  {
    if ( *(_DWORD *)(a3 - 24) != 2 )
      return;
    v20 = *(_QWORD *)(a3 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(a3 - 16) >> 6) & 0xF) != 2 )
      return;
    v20 = a3 - 16 - 8LL * ((v15 >> 2) & 0xF);
  }
  v21 = *(_QWORD *)(v20 + 8);
  if ( v21 )
  {
    sub_B10CB0(v25, v21);
    if ( v25[0] )
    {
      v22 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v22) <= 2 )
      {
        sub_CB6200(a1, " @[", 3u);
      }
      else
      {
        *(_BYTE *)(v22 + 2) = 91;
        *(_WORD *)v22 = 16416;
        *(_QWORD *)(a1 + 32) += 3LL;
      }
      sub_2DF6EF0(v25, a1);
      v23 = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == v23 )
      {
        sub_CB6200(a1, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v23 = 93;
        ++*(_QWORD *)(a1 + 32);
      }
      if ( v25[0] )
        sub_B91220((__int64)v25, v25[0]);
    }
  }
}
