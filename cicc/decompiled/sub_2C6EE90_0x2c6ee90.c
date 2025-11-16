// Function: sub_2C6EE90
// Address: 0x2c6ee90
//
__int64 *__fastcall sub_2C6EE90(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _BYTE *v11; // rsi
  _DWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned int v15; // ecx
  __int64 v16; // r15
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r8
  unsigned int v24; // ecx
  __int64 v25; // rbx
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rdx
  __int64 v30; // rcx
  int v32; // ecx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r9
  int v35; // eax
  __int64 v36; // r10
  int v37; // ecx
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // r9
  int v40; // eax
  __int64 v41; // r12
  int v42; // eax
  int v43; // [rsp+0h] [rbp-90h]
  __int64 v44; // [rsp+0h] [rbp-90h]
  _BYTE *v45; // [rsp+10h] [rbp-80h] BYREF
  __int64 v46; // [rsp+18h] [rbp-78h]
  _BYTE v47[48]; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v48; // [rsp+50h] [rbp-40h]

  v3 = a1;
  v6 = sub_2C6EE00(a3, a2);
  v11 = v47;
  v12 = (_DWORD *)v6;
  v46 = 0x600000000LL;
  LODWORD(v6) = *(_DWORD *)(v6 + 16);
  v45 = v47;
  if ( (_DWORD)v6 )
  {
    sub_2C6DEF0((__int64)&v45, (__int64)(v12 + 2), v7, v8, v9, v10);
    v11 = v45;
  }
  v48 = v12[18];
  *(_QWORD *)&v11[8 * (*v12 >> 6)] |= 1LL << *v12;
  if ( a1[1] != a2[1] )
  {
    do
    {
      v13 = sub_2C6EE00(a3, v3);
      v15 = *(_DWORD *)(v13 + 72);
      v16 = v13;
      v17 = v48;
      if ( v15 < v48 )
      {
        v32 = v15 & 0x3F;
        if ( v32 )
          *(_QWORD *)(*(_QWORD *)(v16 + 8) + 8LL * *(unsigned int *)(v16 + 16) - 8) &= ~(-1LL << v32);
        v33 = *(unsigned int *)(v16 + 16);
        *(_DWORD *)(v16 + 72) = v17;
        v34 = (v17 + 63) >> 6;
        if ( v34 != v33 )
        {
          if ( v34 >= v33 )
          {
            v36 = v34 - v33;
            if ( v34 > *(unsigned int *)(v16 + 20) )
            {
              v44 = v34 - v33;
              sub_C8D5F0(v16 + 8, (const void *)(v16 + 24), v34, 8u, v14, v34);
              v33 = *(unsigned int *)(v16 + 16);
              v36 = v44;
            }
            if ( 8 * v36 )
            {
              v43 = v36;
              memset((void *)(*(_QWORD *)(v16 + 8) + 8 * v33), 0, 8 * v36);
              LODWORD(v33) = *(_DWORD *)(v16 + 16);
              LODWORD(v36) = v43;
            }
            v17 = *(_DWORD *)(v16 + 72);
            *(_DWORD *)(v16 + 16) = v36 + v33;
          }
          else
          {
            *(_DWORD *)(v16 + 16) = (v17 + 63) >> 6;
          }
        }
        v35 = v17 & 0x3F;
        if ( v35 )
          *(_QWORD *)(*(_QWORD *)(v16 + 8) + 8LL * *(unsigned int *)(v16 + 16) - 8) &= ~(-1LL << v35);
      }
      v18 = 0;
      v19 = 8LL * (unsigned int)v46;
      if ( (_DWORD)v46 )
      {
        do
        {
          v20 = (_QWORD *)(v18 + *(_QWORD *)(v16 + 8));
          v21 = *(_QWORD *)&v45[v18];
          v18 += 8;
          *v20 |= v21;
        }
        while ( v18 != v19 );
      }
      v3 = (__int64 *)v3[1];
    }
    while ( v3[1] != a2[1] );
  }
  v22 = sub_2C6EE00(a3, v3);
  v24 = *(_DWORD *)(v22 + 72);
  v25 = v22;
  v26 = v48;
  if ( v24 < v48 )
  {
    v37 = v24 & 0x3F;
    if ( v37 )
      *(_QWORD *)(*(_QWORD *)(v25 + 8) + 8LL * *(unsigned int *)(v25 + 16) - 8) &= ~(-1LL << v37);
    v38 = *(unsigned int *)(v25 + 16);
    *(_DWORD *)(v25 + 72) = v26;
    v39 = (v26 + 63) >> 6;
    if ( v39 != v38 )
    {
      if ( v39 >= v38 )
      {
        v41 = v39 - v38;
        if ( v39 > *(unsigned int *)(v25 + 20) )
        {
          sub_C8D5F0(v25 + 8, (const void *)(v25 + 24), v39, 8u, v23, v39);
          v38 = *(unsigned int *)(v25 + 16);
        }
        if ( 8 * v41 )
        {
          memset((void *)(*(_QWORD *)(v25 + 8) + 8 * v38), 0, 8 * v41);
          LODWORD(v38) = *(_DWORD *)(v25 + 16);
        }
        v42 = *(_DWORD *)(v25 + 72);
        *(_DWORD *)(v25 + 16) = v41 + v38;
        v40 = v42 & 0x3F;
        if ( !v40 )
          goto LABEL_9;
LABEL_32:
        *(_QWORD *)(*(_QWORD *)(v25 + 8) + 8LL * *(unsigned int *)(v25 + 16) - 8) &= ~(-1LL << v40);
        goto LABEL_9;
      }
      *(_DWORD *)(v25 + 16) = (v26 + 63) >> 6;
    }
    v40 = v26 & 0x3F;
    if ( !v40 )
      goto LABEL_9;
    goto LABEL_32;
  }
LABEL_9:
  v27 = 0;
  v28 = 8LL * (unsigned int)v46;
  if ( (_DWORD)v46 )
  {
    do
    {
      v29 = (_QWORD *)(v27 + *(_QWORD *)(v25 + 8));
      v30 = *(_QWORD *)&v45[v27];
      v27 += 8;
      *v29 |= v30;
    }
    while ( v28 != v27 );
  }
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  return v3;
}
