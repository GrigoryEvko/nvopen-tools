// Function: sub_39A9FD0
// Address: 0x39a9fd0
//
void __fastcall sub_39A9FD0(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, unsigned int a5, int a6)
{
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r15
  int i; // r14d
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned int v14; // ebx
  __int64 v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int64 v20; // rdi
  __int64 v21; // r12
  int v22; // r14d
  __int64 v23; // rsi
  unsigned __int64 v24; // rax
  unsigned int v25; // r13d
  __int64 v26; // rsi
  unsigned int v27; // r12d
  __int64 v28; // rax
  int v29; // ebx
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned int v33; // ebx
  __int64 v34; // r12
  int v35; // eax
  int v36; // edx
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // r14
  unsigned int v40; // r15d
  int v41; // esi
  __int64 v42; // rdi
  int v43; // r8d
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // ebx
  int v47; // eax
  int v48; // ecx
  int v49; // [rsp+Ch] [rbp-D4h]
  __int64 v50; // [rsp+10h] [rbp-D0h]
  int v52; // [rsp+20h] [rbp-C0h]
  int v53; // [rsp+24h] [rbp-BCh]
  __int64 v54; // [rsp+28h] [rbp-B8h]
  __int64 *v55; // [rsp+30h] [rbp-B0h]
  __int64 v56; // [rsp+38h] [rbp-A8h]
  int v57; // [rsp+48h] [rbp-98h]
  unsigned int v58; // [rsp+48h] [rbp-98h]
  int v59; // [rsp+48h] [rbp-98h]
  _BYTE *v60; // [rsp+60h] [rbp-80h] BYREF
  __int64 v61; // [rsp+68h] [rbp-78h]
  _BYTE v62[112]; // [rsp+70h] [rbp-70h] BYREF

  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 264LL);
  v8 = *(_QWORD *)(v7 + 560);
  v60 = v62;
  v61 = 0x1000000000LL;
  v9 = *(_QWORD *)(v7 + 552);
  if ( (unsigned __int64)(v8 - v9) > 0x40 )
  {
    sub_16CD150((__int64)&v60, v62, (v8 - v9) >> 2, 4, a5, a6);
    v8 = *(_QWORD *)(v7 + 560);
    v9 = *(_QWORD *)(v7 + 552);
  }
  v10 = v9;
  for ( i = -1; v8 != v10; i -= sub_3946290(*(unsigned int *)(v10 - 4)) )
  {
    v12 = (unsigned int)v61;
    if ( (unsigned int)v61 >= HIDWORD(v61) )
    {
      sub_16CD150((__int64)&v60, v62, 0, 4, a5, a6);
      v12 = (unsigned int)v61;
    }
    v10 += 4;
    *(_DWORD *)&v60[4 * v12] = i;
    LODWORD(v61) = v61 + 1;
  }
  v13 = *((unsigned int *)a2 + 2);
  if ( *(_DWORD *)(a4 + 12) < (unsigned int)v13 )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), v13, 4, a5, a6);
    v13 = *((unsigned int *)a2 + 2);
  }
  v14 = 0;
  v15 = a3;
  v54 = 0;
  v49 = 0;
  v55 = *a2;
  v50 = (__int64)&(*a2)[v13];
  if ( *a2 != (__int64 *)v50 )
  {
    do
    {
      v16 = v54;
      v17 = *(_QWORD *)(*v55 + 104);
      v18 = *(_QWORD *)(*v55 + 96);
      v54 = *v55;
      if ( v16 )
      {
        v19 = *(_QWORD *)(v16 + 96);
        v20 = (v17 - v18) >> 2;
        v21 = (*(_QWORD *)(v16 + 104) - v19) >> 2;
        v53 = v20;
        a5 = v21;
        v22 = v21;
        if ( (unsigned int)v20 <= (unsigned int)v21 )
          a5 = (v17 - v18) >> 2;
        if ( a5 )
        {
          v23 = a5;
          v24 = 0;
          while ( 1 )
          {
            a6 = *(_DWORD *)(v19 + 4 * v24);
            a5 = v24;
            if ( *(_DWORD *)(v18 + 4 * v24) != a6 )
              break;
            v25 = ++v24;
            if ( v23 == v24 )
            {
              if ( v25 >= v20 )
                goto LABEL_18;
              goto LABEL_24;
            }
          }
          if ( v24 >= v20 )
            goto LABEL_18;
          if ( !(_DWORD)v24 )
            goto LABEL_47;
          v25 = v24;
LABEL_24:
          v56 = (unsigned int)(*(_DWORD *)(v15 + 8) - 1);
          v29 = v56;
          v57 = sub_39462B0(*(int *)(*(_QWORD *)v15 + 12 * v56 + 4));
          v30 = sub_39462B0(*(int *)(*(_QWORD *)v15 + 12 * v56));
          v31 = v56;
          a5 = v30 + v57;
          if ( v25 != (_DWORD)v21 )
          {
            v58 = v25;
            v32 = *(_QWORD *)v15;
            v33 = a5;
            do
            {
              ++v25;
              v34 = 12 * v31;
              v35 = sub_39462B0(*(int *)(v32 + 12 * v31));
              v32 = *(_QWORD *)v15;
              v36 = v35;
              v37 = *(_QWORD *)v15 + v34;
              v33 -= *(_DWORD *)(v37 + 4) + v36;
              v31 = *(unsigned int *)(v37 + 8);
            }
            while ( v25 != v22 );
            v25 = v58;
            a5 = v33;
            v29 = *(_DWORD *)(v37 + 8);
          }
          v18 = *(_QWORD *)(v54 + 96);
          v53 = (*(_QWORD *)(v54 + 104) - v18) >> 2;
          if ( v25 != v53 )
            goto LABEL_33;
LABEL_29:
          v52 = 0;
          v38 = v49;
          goto LABEL_44;
        }
        if ( v17 == v18 )
        {
LABEL_18:
          v26 = a4;
          v27 = v14;
          v28 = *(unsigned int *)(a4 + 8);
          if ( (unsigned int)v28 >= *(_DWORD *)(a4 + 12) )
            goto LABEL_45;
          goto LABEL_19;
        }
LABEL_47:
        v25 = 0;
        v29 = -1;
      }
      else
      {
        if ( v17 == v18 )
          goto LABEL_18;
        v25 = 0;
        v29 = -1;
        a5 = 0;
        v53 = (v17 - v18) >> 2;
      }
      if ( !v53 )
        goto LABEL_29;
LABEL_33:
      v39 = v15;
      v59 = v29;
      v40 = a5;
      v52 = 0;
      while ( 1 )
      {
        v46 = *(_DWORD *)(v18 + 4LL * v25);
        if ( v46 < 0 )
          v46 = *(_DWORD *)&v60[4 * ~v46];
        v47 = sub_39462B0(v46);
        if ( v40 )
        {
          v41 = -(v47 + v40);
          v42 = v41;
        }
        else
        {
          v42 = 0;
          v41 = 0;
        }
        v40 = sub_39462B0(v42) + v47;
        v52 += v40;
        v44 = *(unsigned int *)(v39 + 8);
        if ( (unsigned int)v44 >= *(_DWORD *)(v39 + 12) )
        {
          sub_16CD150(v39, (const void *)(v39 + 16), 0, 12, v43, a6);
          v44 = *(unsigned int *)(v39 + 8);
        }
        ++v25;
        v45 = *(_QWORD *)v39 + 12 * v44;
        *(_QWORD *)v45 = __PAIR64__(v41, v46);
        *(_DWORD *)(v45 + 8) = v59;
        v59 = *(_DWORD *)(v39 + 8);
        *(_DWORD *)(v39 + 8) = v59 + 1;
        if ( v25 == v53 )
          break;
        v18 = *(_QWORD *)(v54 + 96);
      }
      a5 = v40;
      v38 = v49 + v52;
      v15 = v39;
LABEL_44:
      v48 = v49;
      v49 = v38;
      v26 = a4;
      v27 = v52 + v48 + 1 - a5;
      v28 = *(unsigned int *)(a4 + 8);
      v14 = v27;
      if ( (unsigned int)v28 >= *(_DWORD *)(a4 + 12) )
      {
LABEL_45:
        sub_16CD150(v26, (const void *)(v26 + 16), 0, 4, a5, a6);
        v28 = *(unsigned int *)(v26 + 8);
      }
LABEL_19:
      ++v55;
      *(_DWORD *)(*(_QWORD *)a4 + 4 * v28) = v27;
      ++*(_DWORD *)(a4 + 8);
    }
    while ( (__int64 *)v50 != v55 );
  }
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
}
