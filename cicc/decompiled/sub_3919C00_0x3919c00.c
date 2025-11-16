// Function: sub_3919C00
// Address: 0x3919c00
//
void __fastcall sub_3919C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, const void *a6)
{
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int8 v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r12
  size_t v14; // rdx
  __int64 v15; // rdi
  size_t v16; // r12
  __int64 v17; // r14
  __int64 v18; // rdi
  size_t v19; // r14
  int v20; // edx
  _BYTE *v21; // rdx
  _BYTE *i; // rcx
  const void *v23; // [rsp+0h] [rbp-50h]
  const void *v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25; // [rsp+8h] [rbp-48h]
  _QWORD v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(unsigned int *)(a2 + 24) * ((*(unsigned int *)(a2 + 24) + v7 - 1) / *(unsigned int *)(a2 + 24));
  if ( v8 >= v7 )
  {
    if ( v8 <= v7 )
      goto LABEL_3;
    if ( v8 > *(unsigned int *)(a1 + 12) )
    {
      v25 = *(unsigned int *)(a2 + 24) * ((*(unsigned int *)(a2 + 24) + v7 - 1) / *(unsigned int *)(a2 + 24));
      sub_16CD150(a1, (const void *)(a1 + 16), v8, 1, a5, (int)a6);
      v7 = *(unsigned int *)(a1 + 8);
      v8 = v25;
    }
    v21 = (_BYTE *)(*(_QWORD *)a1 + v7);
    for ( i = (_BYTE *)(v8 + *(_QWORD *)a1); i != v21; ++v21 )
    {
      if ( v21 )
        *v21 = 0;
    }
  }
  *(_DWORD *)(a1 + 8) = v8;
LABEL_3:
  v9 = *(_QWORD *)(a2 + 104);
  v10 = a2 + 96;
  v24 = (const void *)(a1 + 16);
  if ( v9 != a2 + 96 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(v9 + 17) )
          sub_16BD130("only data supported in data sections", 1u);
        v11 = *(_BYTE *)(v9 + 16);
        if ( !v11 )
          break;
        a6 = *(const void **)(v9 + 64);
        if ( v11 != 3 )
        {
          v18 = *(unsigned int *)(a1 + 8);
          v19 = *(unsigned int *)(v9 + 72);
          v20 = *(_DWORD *)(a1 + 8);
          if ( v19 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v18 )
          {
            v23 = *(const void **)(v9 + 64);
            sub_16CD150(a1, v24, v19 + v18, 1, a5, (int)a6);
            v18 = *(unsigned int *)(a1 + 8);
            a6 = v23;
            v20 = *(_DWORD *)(a1 + 8);
          }
          if ( v19 )
          {
            memcpy((void *)(*(_QWORD *)a1 + v18), a6, v19);
            v20 = *(_DWORD *)(a1 + 8);
          }
          LODWORD(v13) = v20 + v19;
          goto LABEL_14;
        }
        sub_38CF290(*(_QWORD *)(v9 + 64), v26);
        v15 = *(unsigned int *)(a1 + 8);
        v16 = v26[0] * *(unsigned __int8 *)(v9 + 56);
        v17 = *(_QWORD *)(v9 + 48);
        if ( v16 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v15 )
        {
          sub_16CD150(a1, v24, v16 + v15, 1, a5, (int)a6);
          v15 = *(unsigned int *)(a1 + 8);
        }
        if ( v16 )
        {
          memset((void *)(*(_QWORD *)a1 + v15), (unsigned __int8)v17, v16);
          *(_DWORD *)(a1 + 8) += v16;
        }
        else
        {
          *(_DWORD *)(a1 + 8) = v15;
        }
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          return;
      }
      if ( *(_DWORD *)(v9 + 64) != 1 )
        sub_16BD130("only byte values supported for alignment", 1u);
      if ( (*(_BYTE *)(v9 + 52) & 1) == 0 )
        v11 = *(_BYTE *)(v9 + 56);
      v12 = *(unsigned int *)(a1 + 8);
      v13 = v12 + *(unsigned int *)(v9 + 68);
      if ( *(unsigned int *)(v9 + 48) * ((*(unsigned int *)(v9 + 48) + v12 - 1) / *(unsigned int *)(v9 + 48)) <= v13 )
        v13 = *(unsigned int *)(v9 + 48) * ((*(unsigned int *)(v9 + 48) + v12 - 1) / *(unsigned int *)(v9 + 48));
      if ( v13 < v12 )
        goto LABEL_14;
      if ( v13 > v12 )
        break;
LABEL_15:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v10 == v9 )
        return;
    }
    if ( v13 > *(unsigned int *)(a1 + 12) )
    {
      sub_16CD150(a1, v24, v13, 1, a5, (int)a6);
      v12 = *(unsigned int *)(a1 + 8);
      v14 = v13 - v12;
      if ( v13 == v12 )
        goto LABEL_14;
    }
    else
    {
      v14 = v13 - v12;
      if ( v13 == v12 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 8) = v13;
        goto LABEL_15;
      }
    }
    memset((void *)(*(_QWORD *)a1 + v12), v11, v14);
    goto LABEL_14;
  }
}
