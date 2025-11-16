// Function: sub_28B7EA0
// Address: 0x28b7ea0
//
void __fastcall sub_28B7EA0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  __int64 i; // r8
  __int64 v5; // rbx
  unsigned __int64 *v6; // r15
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r12
  unsigned __int64 *v9; // r10
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // r11
  unsigned __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int64 v14; // rax
  unsigned int v15; // ecx
  unsigned __int64 *v16; // rax
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rax
  unsigned int v27; // edx
  unsigned __int64 v28; // rax
  unsigned int v29; // ecx
  unsigned __int64 *v30; // rax
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rbx
  unsigned __int64 v33; // r14
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r13
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // r12
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // r12
  unsigned __int64 *v43; // r13
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  unsigned __int64 v48; // [rsp+0h] [rbp-60h]
  __int64 v51; // [rsp+18h] [rbp-48h]
  unsigned __int64 v52; // [rsp+18h] [rbp-48h]
  __int64 v53; // [rsp+20h] [rbp-40h]

  v51 = (a3 - 1) / 2;
  if ( a2 < v51 )
  {
    for ( i = a2; ; i = v5 )
    {
      v5 = 2 * (i + 1);
      v6 = (unsigned __int64 *)(a1 + 48 * (i + 1));
      v7 = *v6;
      v8 = v6[1];
      v9 = (unsigned __int64 *)(a1 + 24 * (v5 - 1));
      v10 = v9[1];
      v11 = *v9;
      if ( *v6 == v8 )
        break;
      v12 = *v6;
      v13 = -1;
      do
      {
        if ( v13 > *(_DWORD *)(v12 + 92) )
          v13 = *(_DWORD *)(v12 + 92);
        v12 += 192LL;
      }
      while ( v8 != v12 );
      if ( v11 != v10 )
        goto LABEL_9;
      v15 = -1;
LABEL_13:
      if ( v15 > v13 )
      {
        v7 = *v9;
        v6 = (unsigned __int64 *)(a1 + 24 * --v5);
      }
LABEL_15:
      v16 = (unsigned __int64 *)(a1 + 24 * i);
      v17 = *v16;
      *v16 = v7;
      v18 = v16[1];
      v19 = v17;
      v16[1] = v6[1];
      v16[2] = v6[2];
      *v6 = 0;
      v6[1] = 0;
      for ( v6[2] = 0; v18 != v19; v19 += 192LL )
      {
        if ( *(_DWORD *)(v19 + 168) > 0x40u )
        {
          v20 = *(_QWORD *)(v19 + 160);
          if ( v20 )
            j_j___libc_free_0_0(v20);
        }
        if ( *(_DWORD *)(v19 + 128) > 0x40u )
        {
          v21 = *(_QWORD *)(v19 + 120);
          if ( v21 )
            j_j___libc_free_0_0(v21);
        }
        if ( (*(_BYTE *)(v19 + 16) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v19 + 24), 8LL * *(unsigned int *)(v19 + 32), 8);
      }
      if ( v17 )
        j_j___libc_free_0(v17);
      if ( v5 >= v51 )
        goto LABEL_30;
    }
    v13 = -1;
    if ( v11 == v10 )
      goto LABEL_15;
LABEL_9:
    v14 = *v9;
    v15 = -1;
    do
    {
      if ( v15 > *(_DWORD *)(v14 + 92) )
        v15 = *(_DWORD *)(v14 + 92);
      v14 += 192LL;
    }
    while ( v14 != v10 );
    goto LABEL_13;
  }
  v5 = a2;
  v6 = (unsigned __int64 *)(a1 + 24 * a2);
LABEL_30:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v5 )
  {
    v5 = 2 * v5 + 1;
    v41 = *v6;
    v42 = v6[1];
    v43 = (unsigned __int64 *)(a1 + 24 * v5);
    *v6 = *v43;
    v6[1] = v43[1];
    v6[2] = v43[2];
    v44 = v41;
    *v43 = 0;
    v43[1] = 0;
    for ( v43[2] = 0; v42 != v44; v44 += 192LL )
    {
      if ( *(_DWORD *)(v44 + 168) > 0x40u )
      {
        v45 = *(_QWORD *)(v44 + 160);
        if ( v45 )
          j_j___libc_free_0_0(v45);
      }
      if ( *(_DWORD *)(v44 + 128) > 0x40u )
      {
        v46 = *(_QWORD *)(v44 + 120);
        if ( v46 )
          j_j___libc_free_0_0(v46);
      }
      if ( (*(_BYTE *)(v44 + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v44 + 24), 8LL * *(unsigned int *)(v44 + 32), 8);
    }
    if ( v41 )
      j_j___libc_free_0(v41);
    v6 = (unsigned __int64 *)(a1 + 24 * v5);
  }
  v22 = *a4;
  v23 = a4[1];
  *a4 = 0;
  a4[1] = 0;
  v52 = v22;
  v24 = a4[2];
  a4[2] = 0;
  v48 = v24;
  v53 = (v5 - 1) / 2;
  if ( v5 > a2 )
  {
    while ( 1 )
    {
      v6 = (unsigned __int64 *)(a1 + 24 * v53);
      v25 = v6[1];
      if ( *v6 == v25 )
      {
        if ( v52 == v23 )
        {
          v6 = (unsigned __int64 *)(a1 + 24 * v5);
          break;
        }
        v27 = -1;
      }
      else
      {
        v26 = *v6;
        v27 = -1;
        do
        {
          if ( v27 > *(_DWORD *)(v26 + 92) )
            v27 = *(_DWORD *)(v26 + 92);
          v26 += 192LL;
        }
        while ( v25 != v26 );
        if ( v52 == v23 )
        {
          v29 = -1;
          goto LABEL_43;
        }
      }
      v28 = v52;
      v29 = -1;
      do
      {
        if ( v29 > *(_DWORD *)(v28 + 92) )
          v29 = *(_DWORD *)(v28 + 92);
        v28 += 192LL;
      }
      while ( v23 != v28 );
LABEL_43:
      v30 = (unsigned __int64 *)(a1 + 24 * v5);
      v31 = *v30;
      v32 = v30[1];
      if ( v29 <= v27 )
      {
        v6 = v30;
        break;
      }
      *v30 = *v6;
      v33 = v31;
      v30[1] = v6[1];
      v30[2] = v6[2];
      *v6 = 0;
      v6[1] = 0;
      for ( v6[2] = 0; v33 != v32; v33 += 192LL )
      {
        if ( *(_DWORD *)(v33 + 168) > 0x40u )
        {
          v34 = *(_QWORD *)(v33 + 160);
          if ( v34 )
            j_j___libc_free_0_0(v34);
        }
        if ( *(_DWORD *)(v33 + 128) > 0x40u )
        {
          v35 = *(_QWORD *)(v33 + 120);
          if ( v35 )
            j_j___libc_free_0_0(v35);
        }
        if ( (*(_BYTE *)(v33 + 16) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v33 + 24), 8LL * *(unsigned int *)(v33 + 32), 8);
      }
      if ( v31 )
        j_j___libc_free_0(v31);
      v5 = v53;
      if ( a2 >= v53 )
        break;
      v53 = (v53 - 1) / 2;
    }
  }
  v36 = *v6;
  v37 = v6[1];
  v6[1] = v23;
  *v6 = v52;
  v38 = v36;
  for ( v6[2] = v48; v37 != v38; v38 += 192LL )
  {
    if ( *(_DWORD *)(v38 + 168) > 0x40u )
    {
      v39 = *(_QWORD *)(v38 + 160);
      if ( v39 )
        j_j___libc_free_0_0(v39);
    }
    if ( *(_DWORD *)(v38 + 128) > 0x40u )
    {
      v40 = *(_QWORD *)(v38 + 120);
      if ( v40 )
        j_j___libc_free_0_0(v40);
    }
    if ( (*(_BYTE *)(v38 + 16) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v38 + 24), 8LL * *(unsigned int *)(v38 + 32), 8);
  }
  if ( v36 )
    j_j___libc_free_0(v36);
}
