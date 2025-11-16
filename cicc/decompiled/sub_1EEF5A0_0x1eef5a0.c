// Function: sub_1EEF5A0
// Address: 0x1eef5a0
//
char **__fastcall sub_1EEF5A0(char **a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v4; // rax
  char *v5; // r8
  unsigned int v6; // edx
  int v7; // r9d
  char *v8; // r14
  char *v9; // rdx
  int v10; // ecx
  unsigned __int64 v11; // rax
  unsigned int v12; // r10d
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned int v15; // r14d
  unsigned int v16; // r15d
  unsigned __int64 v17; // rax
  int v18; // r9d
  unsigned int v19; // ebx
  _QWORD *v20; // rax
  unsigned int v21; // edx
  unsigned int v22; // eax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // [rsp+4h] [rbp-4Ch]
  unsigned int v27; // [rsp+4h] [rbp-4Ch]
  char *v28; // [rsp+8h] [rbp-48h]
  char *v29; // [rsp+10h] [rbp-40h]
  int v30; // [rsp+10h] [rbp-40h]
  char v31; // [rsp+18h] [rbp-38h]
  char v32; // [rsp+18h] [rbp-38h]
  char v33; // [rsp+18h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  *((_DWORD *)a1 + 4) = 0;
  v2 = *(_DWORD *)(a2 + 40);
  if ( !v2 )
    return a1;
  v26 = *(_DWORD *)(a2 + 40);
  v29 = (char *)((v2 + 63) >> 6);
  v4 = malloc(8LL * (_QWORD)v29);
  v5 = v29;
  v6 = (v2 + 63) >> 6;
  LOBYTE(v7) = v26;
  v8 = (char *)v4;
  if ( !v4 )
  {
    if ( 8LL * (_QWORD)v29 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v5 = (char *)((v2 + 63) >> 6);
      LOBYTE(v7) = v26;
      v10 = (_DWORD)a1[2] & 0x3F;
      v12 = (unsigned int)(*((_DWORD *)a1 + 4) + 63) >> 6;
      v11 = v12;
    }
    else
    {
      v25 = malloc(1u);
      LOBYTE(v7) = v26;
      v5 = (char *)((v2 + 63) >> 6);
      v6 = (v2 + 63) >> 6;
      if ( v25 )
      {
        v8 = (char *)v25;
        goto LABEL_4;
      }
      sub_16BD1C0("Allocation failed", 1u);
      LOBYTE(v7) = v26;
      v5 = (char *)((v2 + 63) >> 6);
      v10 = (_DWORD)a1[2] & 0x3F;
      v12 = (unsigned int)(*((_DWORD *)a1 + 4) + 63) >> 6;
      v11 = v12;
    }
    *a1 = 0;
    a1[1] = v5;
    if ( (unsigned __int64)v5 <= v11 )
    {
      if ( !v10 )
        goto LABEL_28;
      goto LABEL_27;
    }
    v9 = &v5[-v11];
    if ( v5 == (char *)v11 )
    {
LABEL_7:
      if ( !v10 )
      {
LABEL_8:
        v32 = v7;
        memset(v8, 0, 8LL * (_QWORD)v5);
        LOBYTE(v7) = v32;
        goto LABEL_9;
      }
LABEL_27:
      *(_QWORD *)&v8[8 * v12 - 8] &= ~(-1LL << v10);
      v5 = a1[1];
      v8 = *a1;
LABEL_28:
      if ( !v5 )
        goto LABEL_9;
      goto LABEL_8;
    }
LABEL_6:
    v27 = v12;
    v28 = v5;
    v30 = v10;
    v31 = v7;
    memset(&v8[8 * v11], 0, 8LL * (_QWORD)v9);
    v12 = v27;
    v5 = v28;
    v10 = v30;
    LOBYTE(v7) = v31;
    goto LABEL_7;
  }
LABEL_4:
  *a1 = v8;
  a1[1] = v5;
  if ( v6 )
  {
    v9 = v5;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    goto LABEL_6;
  }
LABEL_9:
  v13 = *((_DWORD *)a1 + 4);
  if ( v2 > v13 )
  {
    v14 = (unsigned __int64)a1[1];
    v15 = (v13 + 63) >> 6;
    if ( v14 > v15 )
    {
      v24 = v14 - v15;
      if ( v24 )
      {
        v33 = v7;
        memset(&(*a1)[8 * v15], 0, 8 * v24);
        v13 = *((_DWORD *)a1 + 4);
        LOBYTE(v7) = v33;
      }
    }
    if ( (v13 & 0x3F) != 0 )
    {
      *(_QWORD *)&(*a1)[8 * v15 - 8] &= ~(-1LL << (v13 & 0x3F));
      v13 = *((_DWORD *)a1 + 4);
    }
  }
  *((_DWORD *)a1 + 4) = v2;
  if ( v2 < v13 )
  {
    v16 = (v2 + 63) >> 6;
    v17 = (unsigned __int64)a1[1];
    if ( v17 > v16 )
    {
      v23 = v17 - v16;
      if ( v23 )
      {
        memset(&(*a1)[8 * v16], 0, 8 * v23);
        v7 = *((_DWORD *)a1 + 4);
      }
    }
    v18 = v7 & 0x3F;
    if ( v18 )
      *(_QWORD *)&(*a1)[8 * v16 - 8] &= ~(-1LL << v18);
  }
  v19 = *(_DWORD *)(a2 + 40);
  if ( v19 )
  {
    v20 = *a1;
    if ( v19 >> 6 )
    {
      *v20 = -1;
      v21 = 64;
      do
      {
        *(_QWORD *)&(*a1)[8 * ((v21 - 64) >> 6)] = -1;
        v22 = v21;
        v21 += 64;
      }
      while ( v19 >= v21 );
      if ( v19 > v22 )
        *(_QWORD *)&(*a1)[8 * (v22 >> 6)] |= (1LL << v19) - 1;
    }
    else
    {
      *v20 |= (1LL << v19) - 1;
    }
  }
  return a1;
}
