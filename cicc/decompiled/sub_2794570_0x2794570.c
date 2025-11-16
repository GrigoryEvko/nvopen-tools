// Function: sub_2794570
// Address: 0x2794570
//
__int64 __fastcall sub_2794570(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // r15
  __int64 *v6; // r13
  unsigned int v7; // eax
  __int64 v8; // r11
  __int64 *v9; // rdx
  __int64 *v10; // r8
  __int64 *i; // r15
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 *v17; // r13
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 *v21; // r13
  int v22; // eax
  int v23; // eax
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // r8
  __int64 *v28; // r14
  int v29; // eax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  int v38; // eax
  bool v39; // al
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  int v43; // r12d
  __int64 v44; // rax
  int v45; // [rsp+Ch] [rbp-124h]
  int v46; // [rsp+Ch] [rbp-124h]
  __int64 v47; // [rsp+10h] [rbp-120h]
  unsigned int v48; // [rsp+10h] [rbp-120h]
  const void *v49; // [rsp+20h] [rbp-110h]
  __int64 v50; // [rsp+28h] [rbp-108h]
  int v51; // [rsp+28h] [rbp-108h]
  __int64 *v52; // [rsp+28h] [rbp-108h]
  unsigned int v53; // [rsp+28h] [rbp-108h]
  int v54; // [rsp+28h] [rbp-108h]
  unsigned __int64 v55; // [rsp+30h] [rbp-100h] BYREF
  unsigned int v56; // [rsp+38h] [rbp-F8h]
  __int64 v57; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+48h] [rbp-E8h]
  __int64 v59; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v60; // [rsp+58h] [rbp-D8h]
  __int64 *v61; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v62; // [rsp+98h] [rbp-98h]
  _BYTE v63[144]; // [rsp+A0h] [rbp-90h] BYREF

  *(_QWORD *)(a1 + 16) = a1 + 32;
  v49 = (const void *)(a1 + 32);
  *(_DWORD *)a1 = -3;
  *(_BYTE *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  v5 = *((_QWORD *)a3 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = (__int64 *)a3;
  v50 = sub_B43CC0((__int64)a3);
  v7 = sub_AE43F0(v50, v5);
  v8 = v50;
  v57 = 0;
  v9 = &v59;
  v58 = 1;
  do
  {
    *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != (__int64 *)&v61 );
  v56 = v7;
  v61 = (__int64 *)v63;
  v62 = 0x400000000LL;
  if ( v7 > 0x40 )
  {
    v48 = v7;
    sub_C43690((__int64)&v55, 0, 0);
    v7 = v48;
    v8 = v50;
  }
  else
  {
    v55 = 0;
  }
  if ( (unsigned __int8)sub_B4DE70(a3, v8, v7, &v57, &v55) )
  {
    v21 = (__int64 *)sub_BD5C60((__int64)a3);
    v22 = *a3;
    *(_QWORD *)(a1 + 8) = 0;
    *(_DWORD *)a1 = v22 - 29;
    v47 = a1 + 16;
    v23 = sub_2792F80(a2, *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)]);
    v25 = *(unsigned int *)(a1 + 24);
    if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
    {
      v54 = v23;
      sub_C8D5F0(v47, v49, v25 + 1, 4u, v25 + 1, v24);
      v25 = *(unsigned int *)(a1 + 24);
      v23 = v54;
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v25) = v23;
    v26 = (unsigned int)v62;
    v27 = v61;
    ++*(_DWORD *)(a1 + 24);
    v28 = v27;
    v52 = &v27[3 * v26];
    if ( v52 != v27 )
    {
      do
      {
        v29 = sub_2792F80(a2, *v28);
        v32 = *(unsigned int *)(a1 + 24);
        if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
        {
          v46 = v29;
          sub_C8D5F0(v47, v49, v32 + 1, 4u, v30, v31);
          v32 = *(unsigned int *)(a1 + 24);
          v29 = v46;
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v32) = v29;
        ++*(_DWORD *)(a1 + 24);
        v33 = sub_ACCFD0(v21, (__int64)(v28 + 1));
        v34 = sub_2792F80(a2, v33);
        v37 = *(unsigned int *)(a1 + 24);
        if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
        {
          v45 = v34;
          sub_C8D5F0(v47, v49, v37 + 1, 4u, v35, v36);
          v37 = *(unsigned int *)(a1 + 24);
          v34 = v45;
        }
        v28 += 3;
        *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v37) = v34;
        ++*(_DWORD *)(a1 + 24);
      }
      while ( v52 != v28 );
    }
    v16 = v56;
    if ( v56 <= 0x40 )
    {
      v39 = v55 == 0;
    }
    else
    {
      v53 = v56;
      v38 = sub_C444A0((__int64)&v55);
      v16 = v53;
      v39 = v53 == v38;
    }
    if ( !v39 )
    {
      v40 = sub_ACCFD0(v21, (__int64)&v55);
      v43 = sub_2792F80(a2, v40);
      v44 = *(unsigned int *)(a1 + 24);
      if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        sub_C8D5F0(v47, v49, v44 + 1, 4u, v41, v42);
        v44 = *(unsigned int *)(a1 + 24);
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v44) = v43;
      v16 = v56;
      ++*(_DWORD *)(a1 + 24);
    }
  }
  else
  {
    *(_DWORD *)a1 = *a3 - 29;
    *(_QWORD *)(a1 + 8) = *((_QWORD *)a3 + 9);
    if ( (a3[7] & 0x40) != 0 )
    {
      v10 = (__int64 *)*((_QWORD *)a3 - 1);
      v6 = &v10[4 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    }
    else
    {
      v10 = (__int64 *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    }
    for ( i = v10; v6 != i; ++*(_DWORD *)(a1 + 24) )
    {
      v12 = sub_2792F80(a2, *i);
      v15 = *(unsigned int *)(a1 + 24);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        v51 = v12;
        sub_C8D5F0(a1 + 16, v49, v15 + 1, 4u, v13, v14);
        v15 = *(unsigned int *)(a1 + 24);
        v12 = v51;
      }
      i += 4;
      *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v15) = v12;
    }
    v16 = v56;
  }
  if ( v16 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  v17 = v61;
  v18 = (unsigned __int64)&v61[3 * (unsigned int)v62];
  if ( v61 != (__int64 *)v18 )
  {
    do
    {
      v18 -= 24LL;
      if ( *(_DWORD *)(v18 + 16) > 0x40u )
      {
        v19 = *(_QWORD *)(v18 + 8);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
    }
    while ( v17 != (__int64 *)v18 );
    v18 = (unsigned __int64)v61;
  }
  if ( (_BYTE *)v18 != v63 )
    _libc_free(v18);
  if ( (v58 & 1) == 0 )
    sub_C7D6A0(v59, 16LL * v60, 8);
  return a1;
}
