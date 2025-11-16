// Function: sub_2100AA0
// Address: 0x2100aa0
//
__int64 __fastcall sub_2100AA0(_QWORD *a1, int a2, char a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r14
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax
  unsigned int v11; // r12d
  __int64 v12; // rbx
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // r10
  unsigned int v16; // r12d
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  _QWORD *v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // r9
  __int64 v27; // r10
  unsigned __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rbx
  __int64 *i; // r13
  int v32; // r15d
  __int64 v33; // rax
  unsigned int v34; // r8d
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rcx
  _QWORD *v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // [rsp+0h] [rbp-50h]
  int v41; // [rsp+8h] [rbp-48h]
  unsigned int v42; // [rsp+Ch] [rbp-44h]
  __int64 v43; // [rsp+10h] [rbp-40h]
  __int64 v44; // [rsp+10h] [rbp-40h]
  _QWORD *v46; // [rsp+18h] [rbp-38h]
  __int64 v47; // [rsp+18h] [rbp-38h]

  v6 = a2 & 0x7FFFFFFF;
  v42 = a2 & 0x7FFFFFFF;
  v9 = sub_1E6B9A0(
         a1[3],
         *(_QWORD *)(*(_QWORD *)(a1[3] + 24LL) + 16 * v6) & 0xFFFFFFFFFFFFFFF8LL,
         (unsigned __int8 *)byte_3F871B3,
         0,
         a5,
         a6);
  v10 = a1[5];
  v11 = v9 & 0x7FFFFFFF;
  v12 = v9 & 0x7FFFFFFF;
  if ( v10 )
  {
    v13 = *(_QWORD *)(v10 + 312);
    v14 = *(_DWORD *)(v13 + 4 * v6);
    if ( !v14 )
      v14 = a2;
    v11 = v9 & 0x7FFFFFFF;
    v12 = v11;
    *(_DWORD *)(v13 + 4LL * v11) = v14;
  }
  v15 = a1[4];
  v16 = v11 + 1;
  v17 = *(unsigned int *)(v15 + 408);
  if ( v16 <= (unsigned int)v17 )
    goto LABEL_6;
  v22 = v16;
  if ( v16 < v17 )
  {
    *(_DWORD *)(v15 + 408) = v16;
    goto LABEL_6;
  }
  if ( v16 <= v17 )
  {
LABEL_6:
    v18 = *(_QWORD *)(v15 + 400);
    goto LABEL_7;
  }
  if ( v16 > (unsigned __int64)*(unsigned int *)(v15 + 412) )
  {
    v41 = v9;
    v40 = a1[4];
    sub_16CD150(v15 + 400, (const void *)(v15 + 416), v16, 8, v8, v9);
    v15 = v40;
    v9 = v41;
    v22 = v16;
    v17 = *(unsigned int *)(v40 + 408);
  }
  v18 = *(_QWORD *)(v15 + 400);
  v23 = *(_QWORD *)(v15 + 416);
  v24 = (_QWORD *)(v18 + 8 * v22);
  v25 = (_QWORD *)(v18 + 8 * v17);
  if ( v24 != v25 )
  {
    do
      *v25++ = v23;
    while ( v24 != v25 );
    v18 = *(_QWORD *)(v15 + 400);
  }
  *(_DWORD *)(v15 + 408) = v16;
LABEL_7:
  v43 = v15;
  *(_QWORD *)(v18 + 8 * v12) = sub_1DBA290(v9);
  v19 = *(_QWORD *)(*(_QWORD *)(v43 + 400) + 8 * v12);
  v20 = a1[1];
  if ( v20 && *(float *)(v20 + 116) == INFINITY )
    *(_DWORD *)(v19 + 116) = 2139095040;
  if ( a3 )
  {
    v26 = a1[4];
    v27 = 8 * v6;
    v28 = *(unsigned int *)(v26 + 408);
    if ( v42 < (unsigned int)v28 )
    {
      v29 = *(_QWORD *)(*(_QWORD *)(v26 + 400) + 8 * v6);
      if ( v29 )
      {
LABEL_22:
        v30 = *(_QWORD *)(v29 + 104);
        for ( i = (__int64 *)(v26 + 296); v30; v30 = *(_QWORD *)(v30 + 104) )
        {
          v32 = *(_DWORD *)(v30 + 112);
          v33 = sub_145CBF0(i, 120, 16);
          *(_QWORD *)(v33 + 8) = 0x200000000LL;
          *(_QWORD *)v33 = v33 + 16;
          *(_QWORD *)(v33 + 64) = v33 + 80;
          *(_QWORD *)(v33 + 72) = 0x200000000LL;
          *(_QWORD *)(v33 + 96) = 0;
          *(_DWORD *)(v33 + 112) = v32;
          *(_QWORD *)(v33 + 104) = *(_QWORD *)(v19 + 104);
          *(_QWORD *)(v19 + 104) = v33;
        }
        return v19;
      }
    }
    v34 = v42 + 1;
    if ( (unsigned int)v28 < v42 + 1 )
    {
      v36 = v34;
      if ( v34 < v28 )
      {
        *(_DWORD *)(v26 + 408) = v34;
      }
      else if ( v34 > v28 )
      {
        if ( v34 > (unsigned __int64)*(unsigned int *)(v26 + 412) )
        {
          v44 = a1[4];
          v47 = v34;
          sub_16CD150(v26 + 400, (const void *)(v26 + 416), v34, 8, v34, v26);
          v26 = v44;
          v27 = 8 * v6;
          v34 = v42 + 1;
          v36 = v47;
          v28 = *(unsigned int *)(v44 + 408);
        }
        v35 = *(_QWORD *)(v26 + 400);
        v37 = *(_QWORD *)(v26 + 416);
        v38 = (_QWORD *)(v35 + 8 * v36);
        v39 = (_QWORD *)(v35 + 8 * v28);
        if ( v38 != v39 )
        {
          do
            *v39++ = v37;
          while ( v38 != v39 );
          v35 = *(_QWORD *)(v26 + 400);
        }
        *(_DWORD *)(v26 + 408) = v34;
        goto LABEL_27;
      }
    }
    v35 = *(_QWORD *)(v26 + 400);
LABEL_27:
    v46 = (_QWORD *)v26;
    *(_QWORD *)(v27 + v35) = sub_1DBA290(a2);
    v29 = *(_QWORD *)(v46[50] + 8 * v6);
    sub_1DBB110(v46, v29);
    v26 = a1[4];
    goto LABEL_22;
  }
  return v19;
}
