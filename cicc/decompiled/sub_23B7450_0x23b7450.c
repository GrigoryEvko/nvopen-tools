// Function: sub_23B7450
// Address: 0x23b7450
//
void __fastcall sub_23B7450(__int64 a1, _QWORD *a2, void (__fastcall *a3)(__int64, __int64, __int64), __int64 a4)
{
  _QWORD *v4; // r12
  __int64 v5; // rax
  _BYTE **v6; // rbx
  __int64 v7; // r15
  const void *v8; // r14
  size_t v9; // r13
  int v10; // eax
  int v11; // eax
  _QWORD *v12; // rax
  __int64 *v13; // rsi
  __int64 v14; // rax
  const void *v15; // r13
  size_t v16; // r14
  int v17; // eax
  int v18; // eax
  __int64 v19; // rax
  const void *v20; // r14
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  size_t v24; // r14
  int v25; // eax
  int v26; // eax
  _QWORD *v27; // rax
  size_t v28; // rdx
  __int64 *v29; // r13
  __int64 *v30; // r12
  __int64 v31; // rdx
  const void *v32; // r14
  size_t v33; // r13
  int v34; // eax
  int v35; // eax
  __int64 *v36; // rax
  size_t v37; // r14
  __int64 v38; // r13
  int v39; // eax
  __int64 v40; // r13
  int v41; // eax
  _QWORD *v42; // rax
  __int64 v43; // rax
  size_t v44; // r14
  int v45; // eax
  int v46; // eax
  _QWORD *v47; // rax
  const void *v48; // r15
  __int64 v49; // r14
  int v50; // eax
  int v51; // eax
  __int64 *v52; // r12
  __int64 *v53; // rbx
  __int64 v54; // rdx
  __int64 v55; // [rsp+8h] [rbp-C8h]
  __int64 v56; // [rsp+10h] [rbp-C0h]
  __int64 *v59; // [rsp+28h] [rbp-A8h]
  __int64 *v60; // [rsp+40h] [rbp-90h]
  _BYTE **v61; // [rsp+48h] [rbp-88h]
  size_t v62; // [rsp+50h] [rbp-80h]
  _QWORD *v63; // [rsp+50h] [rbp-80h]
  __int64 v64; // [rsp+58h] [rbp-78h]
  __int64 v65; // [rsp+58h] [rbp-78h]
  const void *v66; // [rsp+58h] [rbp-78h]
  const void *v67; // [rsp+58h] [rbp-78h]
  const void *v68; // [rsp+58h] [rbp-78h]
  size_t v69; // [rsp+58h] [rbp-78h]
  __int64 *v70; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v71; // [rsp+68h] [rbp-68h]
  __int64 *v72; // [rsp+70h] [rbp-60h]
  _QWORD *v73; // [rsp+80h] [rbp-50h] BYREF
  size_t v74; // [rsp+88h] [rbp-48h]
  _QWORD v75[8]; // [rsp+90h] [rbp-40h] BYREF

  v4 = a2;
  v59 = (__int64 *)(a1 + 24);
  v61 = *(_BYTE ***)(a1 + 8);
  v5 = *a2;
  v60 = a2 + 3;
  v6 = *(_BYTE ***)a1;
  v7 = *a2 + 32LL;
  v55 = a2[1];
  v70 = 0;
  v71 = 0;
  v72 = 0;
  if ( v55 == v5 )
  {
    if ( v6 == v61 )
      return;
  }
  else
  {
    do
    {
      v15 = *(const void **)(v7 - 32);
      v56 = v7;
      v16 = *(_QWORD *)(v7 - 24);
      v64 = *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32);
      v17 = sub_C92610();
      v18 = sub_C92860(v59, v15, v16, v17);
      if ( v18 == -1 )
        v19 = *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32);
      else
        v19 = *(_QWORD *)(a1 + 24) + 8LL * v18;
      if ( v19 == v64 )
      {
        v8 = *(const void **)(v7 - 32);
        v9 = *(_QWORD *)(v7 - 24);
        v10 = sub_C92610();
        v11 = sub_C92860(v60, v8, v9, v10);
        if ( v11 == -1 )
          v12 = (_QWORD *)(v4[3] + 8LL * *((unsigned int *)v4 + 8));
        else
          v12 = (_QWORD *)(v4[3] + 8LL * v11);
        v13 = v71;
        v14 = *v12 + 8LL;
        v73 = (_QWORD *)v14;
        if ( v71 == v72 )
        {
          sub_23B72C0((__int64)&v70, v71, &v73);
        }
        else
        {
          if ( v71 )
          {
            *v71 = v14;
            v13 = v71;
          }
          v71 = v13 + 1;
        }
      }
      else
      {
        while ( v61 != v6 )
        {
          v28 = (size_t)v6[1];
          if ( v28 == *(_QWORD *)(v7 - 24) && (!v28 || !memcmp(*v6, *(const void **)(v7 - 32), v28)) )
            break;
          v73 = v75;
          sub_23AEDD0((__int64 *)&v73, *v6, (__int64)&v6[1][(_QWORD)*v6]);
          v20 = v73;
          v62 = v74;
          v65 = v4[3] + 8LL * *((unsigned int *)v4 + 8);
          v21 = sub_C92610();
          v22 = sub_C92860(v60, v20, v62, v21);
          if ( v22 == -1 )
            v23 = v4[3] + 8LL * *((unsigned int *)v4 + 8);
          else
            v23 = v4[3] + 8LL * v22;
          if ( v23 == v65 )
          {
            v24 = (size_t)v6[1];
            v66 = *v6;
            v25 = sub_C92610();
            v26 = sub_C92860(v59, v66, v24, v25);
            if ( v26 == -1 )
              v27 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32));
            else
              v27 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * v26);
            a3(a4, *v27 + 8LL, 0);
          }
          if ( v73 != v75 )
            j_j___libc_free_0((unsigned __int64)v73);
          v6 += 4;
        }
        v29 = v71;
        if ( v70 != v71 )
        {
          v63 = v4;
          v30 = v70;
          do
          {
            v31 = *v30++;
            a3(a4, 0, v31);
          }
          while ( v29 != v30 );
          v4 = v63;
          if ( v70 != v71 )
            v71 = v70;
        }
        v32 = *(const void **)(v7 - 32);
        v33 = *(_QWORD *)(v7 - 24);
        v34 = sub_C92610();
        v35 = sub_C92860(v60, v32, v33, v34);
        if ( v35 == -1 )
          v36 = (__int64 *)(v4[3] + 8LL * *((unsigned int *)v4 + 8));
        else
          v36 = (__int64 *)(v4[3] + 8LL * v35);
        v37 = *(_QWORD *)(v7 - 24);
        v38 = *v36;
        v67 = *(const void **)(v7 - 32);
        v39 = sub_C92610();
        v40 = v38 + 8;
        v41 = sub_C92860(v59, v67, v37, v39);
        if ( v41 == -1 )
          v42 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32));
        else
          v42 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * v41);
        a3(a4, *v42 + 8LL, v40);
        if ( v61 != v6 )
          v6 += 4;
      }
      v7 += 32;
    }
    while ( v55 != v56 );
    if ( v61 == v6 )
      goto LABEL_58;
  }
  do
  {
    v73 = v75;
    sub_23AEDD0((__int64 *)&v73, *v6, (__int64)&v6[1][(_QWORD)*v6]);
    v48 = v73;
    v69 = v74;
    v49 = v4[3] + 8LL * *((unsigned int *)v4 + 8);
    v50 = sub_C92610();
    v51 = sub_C92860(v60, v48, v69, v50);
    if ( v51 == -1 )
      v43 = v4[3] + 8LL * *((unsigned int *)v4 + 8);
    else
      v43 = v4[3] + 8LL * v51;
    if ( v43 == v49 )
    {
      v44 = (size_t)v6[1];
      v68 = *v6;
      v45 = sub_C92610();
      v46 = sub_C92860(v59, v68, v44, v45);
      if ( v46 == -1 )
        v47 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32));
      else
        v47 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * v46);
      a3(a4, *v47 + 8LL, 0);
    }
    if ( v73 != v75 )
      j_j___libc_free_0((unsigned __int64)v73);
    v6 += 4;
  }
  while ( v61 != v6 );
LABEL_58:
  v52 = v71;
  v53 = v70;
  if ( v70 != v71 )
  {
    do
    {
      v54 = *v53++;
      a3(a4, 0, v54);
    }
    while ( v52 != v53 );
    v52 = v70;
    if ( v70 != v71 )
      v71 = v70;
  }
  if ( v52 )
    j_j___libc_free_0((unsigned __int64)v52);
}
