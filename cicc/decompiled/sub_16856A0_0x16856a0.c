// Function: sub_16856A0
// Address: 0x16856a0
//
int __fastcall sub_16856A0(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // r13
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  char *v11; // r12
  char *v12; // rcx
  char *v13; // r15
  _QWORD *v14; // rdx
  char *v15; // rsi
  int v16; // eax
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // [rsp+8h] [rbp-48h]
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  char *v29; // [rsp+10h] [rbp-40h]
  _QWORD *v30; // [rsp+10h] [rbp-40h]
  char *v31; // [rsp+10h] [rbp-40h]
  char *v32; // [rsp+10h] [rbp-40h]

  v2 = sub_1684C80((unsigned __int64)a1);
  if ( v2 )
  {
    v3 = v2;
    sub_1684B50((pthread_mutex_t **)(*(_QWORD *)(v2 + 24) + 7128LL));
    if ( *(_BYTE *)(v3 + 40) )
    {
      v7 = *(unsigned int *)(v3 + 48);
      v6 = *(_QWORD *)(v3 + 24);
      if ( v7 <= 0x1387 )
        goto LABEL_4;
    }
    else
    {
      v6 = *(_QWORD *)(v3 + 24);
      v7 = *(a1 - 2) - 32LL;
      if ( v7 <= 0x1387 )
      {
LABEL_4:
        v8 = (unsigned int)(v7 >> 3) + 266LL;
        v9 = *(_QWORD *)(v6 + 8 * v8);
        a1[1] = v3;
        *a1 = v9;
        *(_QWORD *)(v6 + 8 * v8) = a1;
        *(_QWORD *)(v3 + 8) += v7;
        return j__pthread_mutex_unlock(*(pthread_mutex_t **)(*(_QWORD *)(v3 + 24) + 7128LL));
      }
    }
    v11 = (char *)(a1 - 4);
    v12 = (char *)*(a1 - 1);
    v13 = (char *)a1 + *(a1 - 2) - 32;
    v14 = (_QWORD *)((char *)(a1 - 4) - v12);
    v15 = v13;
    if ( a1 == (_QWORD *)32 )
    {
      v28 = -(__int64)v12;
      v32 = (char *)MEMORY[0x18];
      sub_16863E0((unsigned int)&unk_4CD28D0, (_DWORD)v13, (_DWORD)v14, (_DWORD)v12, v4, v5);
      v14 = (_QWORD *)v28;
      v12 = v32;
      v15 = v13;
    }
    if ( *(a1 - 4) != -1 )
    {
      v26 = v14;
      v29 = v12;
      sub_16863E0((unsigned int)&unk_4CD28D0, (_DWORD)v15, (_DWORD)v14, (_DWORD)v12, v4, v5);
      v14 = v26;
      v12 = v29;
    }
    *(_QWORD *)(v3 + 8) += *((_QWORD *)v11 + 2);
    v16 = *(_DWORD *)(v6 + 56);
    if ( v16 )
      *(_DWORD *)(v6 + 56) = v16 - 1;
    if ( !v13 )
    {
      v27 = v14;
      v31 = v12;
      sub_16863E0((unsigned int)&unk_4CD28D0, (_DWORD)v15, (_DWORD)v14, (_DWORD)v12, v4, v5);
      v14 = v27;
      v12 = v31;
    }
    v17 = *(_QWORD *)v13;
    if ( *(_QWORD *)v13 != -1 )
    {
      v15 = &v13[*((_QWORD *)v13 + 2)];
      if ( v17 )
        *(_QWORD *)(v17 + 8) = *((_QWORD *)v13 + 1);
      v20 = (_QWORD *)*((_QWORD *)v13 + 1);
      if ( v20 )
        *v20 = *(_QWORD *)v13;
      *(_QWORD *)v13 = -1;
      v21 = *((_QWORD *)v11 + 2) + *((_QWORD *)v13 + 2);
      *((_QWORD *)v11 + 2) = v21;
      *((_QWORD *)v15 + 3) = v21;
    }
    if ( v11 == v12 )
    {
      v30 = v14;
      sub_16863E0((unsigned int)&unk_4CD28D0, (_DWORD)v15, (_DWORD)v14, (_DWORD)v12, v4, v5);
      v14 = v30;
    }
    v18 = *((_QWORD *)v11 + 2);
    if ( *v14 == -1 )
    {
      if ( (int)sub_1683CB0(v18) >= 0 )
      {
        v22 = sub_1683CB0(*((_QWORD *)v11 + 2));
        v23 = v6 + 32 * (v22 + 2LL);
        v24 = 32LL * v22 + v6;
        *((_QWORD *)v11 + 1) = v23;
        *(a1 - 4) = *(_QWORD *)(v24 + 64);
        *(_QWORD *)(v24 + 64) = v11;
        v25 = *(a1 - 4);
        if ( v25 )
          *(_QWORD *)(v25 + 8) = v11;
      }
    }
    else
    {
      v19 = v14[2] + v18;
      v14[2] = v19;
      *((_QWORD *)v15 + 3) = v19;
    }
    return j__pthread_mutex_unlock(*(pthread_mutex_t **)(*(_QWORD *)(v3 + 24) + 7128LL));
  }
  return sub_1688C60(a1, 0);
}
