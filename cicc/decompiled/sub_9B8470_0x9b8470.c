// Function: sub_9B8470
// Address: 0x9b8470
//
__int64 __fastcall sub_9B8470(int a1, char *a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r15
  unsigned int v6; // r8d
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r8
  char *v10; // r13
  __int64 v11; // r10
  __int64 v12; // r9
  int v13; // ebx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  size_t v16; // rdx
  __int64 v17; // rax
  int v18; // eax
  int v19; // edx
  _DWORD *v20; // rcx
  int v21; // ebx
  __int64 v23; // rdx
  int v24; // eax
  signed __int64 v25; // r14
  unsigned __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rax
  int v29; // [rsp+4h] [rbp-4Ch]
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  unsigned __int64 v34; // [rsp+10h] [rbp-40h]
  unsigned __int64 v35; // [rsp+10h] [rbp-40h]
  unsigned __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  if ( a1 == 1 )
  {
    v25 = 4 * a3;
    v26 = *(unsigned int *)(a4 + 12);
    *(_DWORD *)(a4 + 8) = 0;
    v27 = 0;
    LODWORD(v28) = 0;
    if ( v25 >> 2 > v26 )
    {
      sub_C8D5F0(a4, a4 + 16, v25 >> 2, 4);
      v28 = *(unsigned int *)(a4 + 8);
      v27 = 4 * v28;
    }
    if ( v25 )
    {
      memcpy((void *)(*(_QWORD *)a4 + v27), a2, v25);
      LODWORD(v28) = *(_DWORD *)(a4 + 8);
    }
    v6 = 1;
    *(_DWORD *)(a4 + 8) = v28 + (v25 >> 2);
    return v6;
  }
  v5 = a3;
  v6 = 0;
  if ( (int)a3 % a1 )
    return v6;
  v7 = (int)a3 / a1;
  v8 = *(unsigned int *)(a4 + 12);
  *(_DWORD *)(a4 + 8) = 0;
  if ( v7 > v8 )
    sub_C8D5F0(a4, a4 + 16, v7, 4);
  v9 = a1;
  v10 = a2 + 4;
  v11 = a4 + 16;
  v12 = 4LL * a1;
  while ( 1 )
  {
    v13 = *((_DWORD *)v10 - 1);
    if ( v13 >= 0 )
      break;
    v14 = v5;
    if ( v9 <= v5 )
      v14 = v9;
    v15 = 4 * v14;
    if ( v15 )
    {
      v16 = v15 - 4;
      if ( v16 )
      {
        v30 = v11;
        v33 = v12;
        v36 = v9;
        v24 = memcmp(v10, v10 - 4, v16);
        v9 = v36;
        v12 = v33;
        v11 = v30;
        if ( v24 )
          return 0;
      }
    }
    v17 = *(unsigned int *)(a4 + 8);
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v32 = v12;
      v35 = v9;
      v38 = v11;
      sub_C8D5F0(a4, v11, v17 + 1, 4);
      v17 = *(unsigned int *)(a4 + 8);
      v12 = v32;
      v9 = v35;
      v11 = v38;
    }
    *(_DWORD *)(*(_QWORD *)a4 + 4 * v17) = v13;
    ++*(_DWORD *)(a4 + 8);
LABEL_14:
    v10 += v12;
    v5 -= v9;
    if ( !v5 )
      return 1;
  }
  v18 = v13 / a1;
  if ( v13 % a1 )
    return 0;
  if ( a1 <= 1 )
  {
LABEL_23:
    v23 = *(unsigned int *)(a4 + 8);
    if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v29 = v18;
      v31 = v12;
      v34 = v9;
      v37 = v11;
      sub_C8D5F0(a4, v11, v23 + 1, 4);
      v23 = *(unsigned int *)(a4 + 8);
      v18 = v29;
      v12 = v31;
      v9 = v34;
      v11 = v37;
    }
    *(_DWORD *)(*(_QWORD *)a4 + 4 * v23) = v18;
    ++*(_DWORD *)(a4 + 8);
    goto LABEL_14;
  }
  v19 = v13 + 1;
  v20 = v10;
  v21 = a1 + v13;
  while ( *v20 == v19 )
  {
    ++v19;
    ++v20;
    if ( v21 == v19 )
      goto LABEL_23;
  }
  return 0;
}
