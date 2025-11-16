// Function: sub_C666F0
// Address: 0xc666f0
//
void __fastcall sub_C666F0(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  unsigned __int8 *v5; // r12
  unsigned __int8 *v6; // r13
  unsigned __int64 v7; // rbx
  __int64 v8; // rdi
  unsigned int v9; // eax
  unsigned __int64 v10; // rbx
  int v11; // eax
  unsigned __int8 v12; // al
  unsigned int v13; // eax
  __int64 v14; // r8
  __int64 v15; // r15
  size_t v16; // rbx
  __int64 v17; // rdi
  char *v18; // rdx
  unsigned __int64 v19; // rbx
  int v20; // eax
  char v21; // al
  __int64 v22; // rdi
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  char *v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  unsigned __int8 *v27; // [rsp+18h] [rbp-38h]

  v5 = a2;
  if ( !*(_QWORD *)(a1 + 80) )
  {
LABEL_2:
    v6 = &v5[a3];
    if ( v5 >= v6 )
      return;
    while ( 1 )
    {
      v8 = *v5;
      if ( (unsigned __int8)(v8 - 32) <= 0x5Eu )
        break;
      v9 = sub_F03780(v8);
      v10 = v6 - v5;
      if ( v9 > (int)v6 - (int)v5 )
      {
        *(_QWORD *)(a1 + 80) = 0;
        v22 = 0;
        if ( v10 > *(_QWORD *)(a1 + 88) )
        {
          sub_C8D290(a1 + 72, a1 + 96, v6 - v5, 1);
          v22 = *(_QWORD *)(a1 + 80);
        }
        if ( v5 != v6 )
        {
          memcpy((void *)(*(_QWORD *)(a1 + 72) + v22), v5, v6 - v5);
          v22 = *(_QWORD *)(a1 + 80);
        }
        *(_QWORD *)(a1 + 80) = v22 + v10;
        return;
      }
      v7 = v9;
      v11 = sub_CA19D0(v5, v9);
      if ( v11 != -1 )
        *(_DWORD *)(a1 + 56) += v11;
      if ( v7 > 1 )
        goto LABEL_5;
      v12 = *v5;
      if ( *v5 == 10 )
      {
        ++*(_DWORD *)(a1 + 60);
LABEL_17:
        *(_DWORD *)(a1 + 56) = 0;
        goto LABEL_5;
      }
      if ( v12 == 13 )
        goto LABEL_17;
      if ( v12 == 9 )
      {
        v5 += v7;
        *(_DWORD *)(a1 + 56) += -*(_DWORD *)(a1 + 56) & 7;
        if ( v6 <= v5 )
          return;
      }
      else
      {
LABEL_5:
        v5 += v7;
        if ( v6 <= v5 )
          return;
      }
    }
    ++*(_DWORD *)(a1 + 56);
    v7 = 1;
    goto LABEL_5;
  }
  v13 = sub_F03780(**(unsigned __int8 **)(a1 + 72));
  v14 = *(_QWORD *)(a1 + 80);
  v15 = v13;
  v16 = v13 - v14;
  if ( a3 >= v16 )
  {
    v17 = *(_QWORD *)(a1 + 80);
    v27 = &a2[v16];
    if ( (unsigned __int64)v13 > *(_QWORD *)(a1 + 88) )
    {
      v26 = v14;
      sub_C8D290(a1 + 72, a1 + 96, v13, 1);
      v17 = *(_QWORD *)(a1 + 80);
      v14 = v26;
    }
    v18 = *(char **)(a1 + 72);
    if ( a2 != v27 )
    {
      v24 = v14;
      memcpy(&v18[v17], a2, v16);
      v18 = *(char **)(a1 + 72);
      v17 = *(_QWORD *)(a1 + 80);
      v14 = v24;
    }
    v19 = v17 + v16;
    v23 = v14;
    *(_QWORD *)(a1 + 80) = v19;
    v25 = v18;
    v20 = sub_CA19D0(v18, v19);
    if ( v20 != -1 )
      *(_DWORD *)(a1 + 56) += v20;
    if ( v19 > 1 )
      goto LABEL_30;
    v21 = *v25;
    if ( *v25 == 10 )
    {
      ++*(_DWORD *)(a1 + 60);
    }
    else if ( v21 != 13 )
    {
      if ( v21 == 9 )
        *(_DWORD *)(a1 + 56) += -*(_DWORD *)(a1 + 56) & 7;
      goto LABEL_30;
    }
    *(_DWORD *)(a1 + 56) = 0;
LABEL_30:
    *(_QWORD *)(a1 + 80) = 0;
    v5 = v27;
    a3 = v23 + a3 - v15;
    goto LABEL_2;
  }
  if ( a3 + v14 > *(_QWORD *)(a1 + 88) )
  {
    sub_C8D290(a1 + 72, a1 + 96, a3 + v14, 1);
    v14 = *(_QWORD *)(a1 + 80);
  }
  if ( a3 )
  {
    memcpy((void *)(v14 + *(_QWORD *)(a1 + 72)), a2, a3);
    v14 = *(_QWORD *)(a1 + 80);
  }
  *(_QWORD *)(a1 + 80) = v14 + a3;
}
