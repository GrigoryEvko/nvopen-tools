// Function: sub_2406750
// Address: 0x2406750
//
__int64 *__fastcall sub_2406750(_QWORD *a1, unsigned __int64 *a2, __int64 a3)
{
  __int64 *v4; // r15
  _QWORD *v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r14
  _QWORD ***v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD ***v11; // r12
  _QWORD **v12; // rcx
  __int64 v13; // rsi
  _QWORD *v14; // rdi
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 *v22; // r14
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned __int64 v33; // r13
  __int64 v34; // r12
  __int64 v35; // rbx
  _QWORD **v36; // r13
  char v37; // si
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rax
  unsigned __int64 v41; // rcx
  __int64 v42; // r13
  _QWORD **v43; // r10
  int v44; // [rsp+8h] [rbp-B8h]
  _QWORD *v45; // [rsp+10h] [rbp-B0h]
  _QWORD **v46; // [rsp+10h] [rbp-B0h]
  __int64 *v47; // [rsp+18h] [rbp-A8h]
  _QWORD ***v50; // [rsp+38h] [rbp-88h]
  __int64 v51; // [rsp+38h] [rbp-88h]
  _QWORD **v52; // [rsp+38h] [rbp-88h]
  __int64 *v53; // [rsp+40h] [rbp-80h] BYREF
  __int64 v54; // [rsp+48h] [rbp-78h]
  _BYTE v55[112]; // [rsp+50h] [rbp-70h] BYREF

  v4 = sub_2404C10(a1, a2);
  v53 = (__int64 *)v55;
  v54 = 0x800000000LL;
  v5 = (_QWORD *)a2[5];
  if ( v5 == (_QWORD *)a2[6] )
    return v4;
  v47 = v4;
  v6 = 0;
  v7 = a3;
  do
  {
    v8 = (_QWORD ***)sub_2406750(a1, *v5, v7);
    v11 = v8;
    if ( v8 )
    {
      if ( !v6 )
      {
        v6 = (__int64)v8;
        goto LABEL_9;
      }
      v12 = *v8;
      v13 = *(unsigned int *)(v6 + 8);
      v14 = *(_QWORD **)(*(_QWORD *)v6 + 96 * v13 - 96);
      v15 = ***v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v15 != v14[4] )
      {
LABEL_6:
        v16 = (unsigned int)v54;
        v17 = (unsigned int)v54 + 1LL;
        if ( v17 > HIDWORD(v54) )
        {
          sub_C8D5F0((__int64)&v53, v55, v17, 8u, v9, v10);
          v16 = (unsigned int)v54;
        }
        v53[v16] = v6;
        v6 = (__int64)v11;
        LODWORD(v54) = v54 + 1;
        goto LABEL_9;
      }
      v29 = *(_QWORD *)(v15 + 16);
      if ( v29 )
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)(v29 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v30 - 30) <= 0xAu )
            break;
          v29 = *(_QWORD *)(v29 + 8);
          if ( !v29 )
            goto LABEL_39;
        }
LABEL_36:
        if ( !(unsigned __int8)sub_22DB400(v14, *(_QWORD *)(v30 + 40)) )
          goto LABEL_6;
        while ( 1 )
        {
          v29 = *(_QWORD *)(v29 + 8);
          if ( !v29 )
            break;
          v30 = *(_QWORD *)(v29 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v30 - 30) <= 0xAu )
            goto LABEL_36;
        }
        v12 = *v11;
        v13 = *(unsigned int *)(v6 + 8);
      }
LABEL_39:
      v31 = 12LL * *((unsigned int *)v11 + 2);
      v32 = (__int64)&v12[v31];
      v33 = 0xAAAAAAAAAAAAAAABLL * ((v31 * 8) >> 5);
      if ( v33 + v13 > *(unsigned int *)(v6 + 12) )
      {
        v46 = v12;
        v51 = v32;
        sub_23FAFC0(v6, v33 + v13, v32, (__int64)v12, v9, v10);
        v13 = *(unsigned int *)(v6 + 8);
        v12 = v46;
        v32 = v51;
      }
      if ( (_QWORD **)v32 != v12 )
      {
        v50 = v11;
        v34 = v32;
        v45 = v5;
        v35 = *(_QWORD *)v6 + 96 * v13;
        v44 = v33;
        v36 = v12;
        do
        {
          while ( 1 )
          {
            if ( v35 )
            {
              *(_QWORD *)v35 = *v36;
              v37 = *((_BYTE *)v36 + 8);
              *(_DWORD *)(v35 + 24) = 0;
              *(_BYTE *)(v35 + 8) = v37;
              *(_QWORD *)(v35 + 16) = v35 + 32;
              *(_DWORD *)(v35 + 28) = 8;
              if ( *((_DWORD *)v36 + 6) )
                break;
            }
            v36 += 12;
            v35 += 96;
            if ( (_QWORD **)v34 == v36 )
              goto LABEL_47;
          }
          v38 = (__int64)(v36 + 2);
          v39 = v35 + 16;
          v36 += 12;
          v35 += 96;
          sub_23FAD70(v39, v38, v32, (__int64)v12, v9, v10);
        }
        while ( (_QWORD **)v34 != v36 );
LABEL_47:
        v11 = v50;
        v5 = v45;
        LODWORD(v33) = v44;
        LODWORD(v13) = *(_DWORD *)(v6 + 8);
      }
      v40 = *(unsigned int *)(v6 + 792);
      v41 = *(unsigned int *)(v6 + 796);
      *(_DWORD *)(v6 + 8) = v33 + v13;
      v42 = *((unsigned int *)v11 + 198);
      v43 = v11[98];
      v9 = 8 * v42;
      if ( v42 + v40 > v41 )
      {
        v52 = v11[98];
        sub_C8D5F0(v6 + 784, (const void *)(v6 + 800), v42 + v40, 8u, v9, v10);
        v40 = *(unsigned int *)(v6 + 792);
        v9 = 8 * v42;
        v43 = v52;
      }
      if ( v9 )
      {
        memcpy((void *)(*(_QWORD *)(v6 + 784) + 8 * v40), v43, v9);
        LODWORD(v40) = *(_DWORD *)(v6 + 792);
      }
      *(_DWORD *)(v6 + 792) = v40 + v42;
    }
    else if ( v6 )
    {
      v27 = (unsigned int)v54;
      v28 = (unsigned int)v54 + 1LL;
      if ( v28 > HIDWORD(v54) )
      {
        sub_C8D5F0((__int64)&v53, v55, v28, 8u, v9, v10);
        v27 = (unsigned int)v54;
      }
      v53[v27] = v6;
      v6 = 0;
      LODWORD(v54) = v54 + 1;
    }
LABEL_9:
    ++v5;
  }
  while ( (_QWORD *)a2[6] != v5 );
  v18 = v7;
  v19 = v6;
  v20 = (unsigned int)v54;
  v4 = v47;
  if ( v19 )
  {
    if ( (unsigned __int64)(unsigned int)v54 + 1 > HIDWORD(v54) )
    {
      sub_C8D5F0((__int64)&v53, v55, (unsigned int)v54 + 1LL, 8u, v9, v10);
      v20 = (unsigned int)v54;
    }
    v53[v20] = v19;
    v20 = (unsigned int)(v54 + 1);
    LODWORD(v54) = v54 + 1;
  }
  v21 = v53;
  v22 = &v53[v20];
  if ( v53 != v22 )
  {
    do
    {
      while ( 1 )
      {
        v24 = *v21;
        if ( v47 )
          break;
        v25 = *(unsigned int *)(v18 + 8);
        if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
        {
          sub_C8D5F0(v18, (const void *)(v18 + 16), v25 + 1, 8u, v9, v10);
          v25 = *(unsigned int *)(v18 + 8);
        }
        ++v21;
        *(_QWORD *)(*(_QWORD *)v18 + 8 * v25) = v24;
        ++*(_DWORD *)(v18 + 8);
        if ( v22 == v21 )
          goto LABEL_23;
      }
      v23 = *((unsigned int *)v47 + 198);
      if ( v23 + 1 > (unsigned __int64)*((unsigned int *)v47 + 199) )
      {
        sub_C8D5F0((__int64)(v47 + 98), v47 + 100, v23 + 1, 8u, v9, v10);
        v23 = *((unsigned int *)v47 + 198);
      }
      ++v21;
      *(_QWORD *)(v47[98] + 8 * v23) = v24;
      ++*((_DWORD *)v47 + 198);
    }
    while ( v22 != v21 );
LABEL_23:
    v22 = v53;
  }
  if ( v22 != (__int64 *)v55 )
    _libc_free((unsigned __int64)v22);
  return v4;
}
