// Function: sub_BDA2E0
// Address: 0xbda2e0
//
__int64 __fastcall sub_BDA2E0(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  char *v6; // r13
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  unsigned __int8 v10; // cl
  unsigned __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rdi
  unsigned __int8 v14; // r10
  unsigned __int64 v15; // r11
  __int64 v16; // rcx
  unsigned __int8 v17; // di
  unsigned __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // r11
  __int64 v21; // rdx
  __int64 *v22; // rdi
  __int64 v23; // r10
  __int64 *v24; // rax
  bool v25; // r9
  __int64 *v26; // r14
  __int64 v27; // r8
  unsigned __int8 v28; // cl
  unsigned __int64 v29; // r13
  unsigned __int64 v30; // r8
  __int64 v31; // r9
  unsigned __int8 v32; // cl
  __int64 v33; // r14
  unsigned __int8 v34; // r11
  unsigned __int64 v35; // r14
  __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rcx
  __int64 *v39; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a3;
  if ( !a3 )
  {
    v26 = a2;
    goto LABEL_45;
  }
  v39 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    --v4;
    v6 = &a1[8 * (result >> 4)];
    v7 = *((_QWORD *)a1 + 1);
    v8 = *(_QWORD *)v6;
    v9 = *(_QWORD *)(*(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)) + 24LL);
    v10 = *(_BYTE *)(v9 - 16);
    if ( (v10 & 2) != 0 )
      v11 = *(_QWORD *)(v9 - 32);
    else
      v11 = -16 - 8LL * ((v10 >> 2) & 0xF) + v9;
    v12 = *(a2 - 1);
    v13 = *(_QWORD *)(*(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)) + 24LL);
    v14 = *(_BYTE *)(v13 - 16);
    if ( (v14 & 2) == 0 )
    {
      v15 = v13 + -16 - 8LL * ((v14 >> 2) & 0xF);
      if ( v11 < v15 )
        goto LABEL_8;
LABEL_33:
      v33 = *(_QWORD *)(*(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF)) + 24LL);
      v34 = *(_BYTE *)(v33 - 16);
      if ( (v34 & 2) != 0 )
        v35 = *(_QWORD *)(v33 - 32);
      else
        v35 = *(_QWORD *)(*(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF)) + 24LL)
            + -16
            - 8LL * ((v34 >> 2) & 0xF);
      v20 = *(_QWORD *)a1;
      if ( v11 < v35 )
      {
        *(_QWORD *)a1 = v7;
        *((_QWORD *)a1 + 1) = v20;
        v21 = *(a2 - 1);
        goto LABEL_12;
      }
      if ( (*(_BYTE *)(v13 - 16) & 2) != 0 )
      {
        if ( v35 > *(_QWORD *)(v13 - 32) )
          goto LABEL_42;
      }
      else if ( v35 > -16LL - 8 * (unsigned __int64)((v14 >> 2) & 0xF) + v13 )
      {
LABEL_42:
        *(_QWORD *)a1 = v12;
        v21 = v20;
        *(a2 - 1) = v20;
        v7 = *(_QWORD *)a1;
        v20 = *((_QWORD *)a1 + 1);
        goto LABEL_12;
      }
      *(_QWORD *)a1 = v8;
      *(_QWORD *)v6 = v20;
      v7 = *(_QWORD *)a1;
      v20 = *((_QWORD *)a1 + 1);
      v21 = *(a2 - 1);
      goto LABEL_12;
    }
    v15 = *(_QWORD *)(v13 - 32);
    if ( v11 >= v15 )
      goto LABEL_33;
LABEL_8:
    v16 = *(_QWORD *)(*(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF)) + 24LL);
    v17 = *(_BYTE *)(v16 - 16);
    if ( (v17 & 2) != 0 )
      v18 = *(_QWORD *)(v16 - 32);
    else
      v18 = -16 - 8LL * ((v17 >> 2) & 0xF) + v16;
    v19 = *(_QWORD *)a1;
    if ( v15 >= v18 )
    {
      if ( v11 >= v18 )
      {
        *(_QWORD *)a1 = v7;
        v20 = v19;
        *((_QWORD *)a1 + 1) = v19;
        v21 = *(a2 - 1);
      }
      else
      {
        *(_QWORD *)a1 = v12;
        v21 = v19;
        *(a2 - 1) = v19;
        v7 = *(_QWORD *)a1;
        v20 = *((_QWORD *)a1 + 1);
      }
    }
    else
    {
      *(_QWORD *)a1 = v8;
      *(_QWORD *)v6 = v19;
      v7 = *(_QWORD *)a1;
      v20 = *((_QWORD *)a1 + 1);
      v21 = *(a2 - 1);
    }
LABEL_12:
    v22 = v39;
    v23 = *(_QWORD *)(*(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)) + 24LL);
    v24 = a2;
    v25 = (*(_BYTE *)(v23 - 16) & 2) != 0;
    while ( 1 )
    {
      v26 = v22 - 1;
      v27 = *(_QWORD *)(*(_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF)) + 24LL);
      v28 = *(_BYTE *)(v27 - 16);
      v29 = (v28 & 2) != 0 ? *(_QWORD *)(v27 - 32) : v27 + -16 - 8LL * ((v28 >> 2) & 0xF);
      v30 = v25 ? *(_QWORD *)(v23 - 32) : v23 + -16 - 8LL * ((*(_BYTE *)(v23 - 16) >> 2) & 0xF);
      if ( v29 >= v30 )
        break;
LABEL_25:
      v20 = *v22++;
    }
    for ( --v24; ; --v24 )
    {
      v31 = *(_QWORD *)(*(_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)) + 24LL);
      v32 = *(_BYTE *)(v31 - 16);
      if ( (v32 & 2) == 0 )
        break;
      if ( v30 >= *(_QWORD *)(v31 - 32) )
        goto LABEL_23;
LABEL_20:
      v21 = *(v24 - 1);
    }
    if ( v30 < -16LL - 8 * (unsigned __int64)((v32 >> 2) & 0xF) + v31 )
      goto LABEL_20;
LABEL_23:
    if ( v24 > v26 )
    {
      *(v22 - 1) = v21;
      *v24 = v20;
      v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 - 32LL * (*(_DWORD *)(*(_QWORD *)a1 + 4LL) & 0x7FFFFFF)) + 24LL);
      v21 = *(v24 - 1);
      v25 = (*(_BYTE *)(v23 - 16) & 2) != 0;
      goto LABEL_25;
    }
    sub_BDA2E0(v22 - 1, a2, v4);
    result = (char *)v26 - a1;
    if ( (char *)v26 - a1 > 128 )
    {
      if ( v4 )
      {
        a2 = v22 - 1;
        continue;
      }
LABEL_45:
      v36 = result >> 3;
      v37 = ((result >> 3) - 2) >> 1;
      sub_BD9F80((__int64)a1, v37, result >> 3, *(_QWORD *)&a1[8 * v37]);
      do
      {
        --v37;
        sub_BD9F80((__int64)a1, v37, v36, *(_QWORD *)&a1[8 * v37]);
      }
      while ( v37 );
      do
      {
        v38 = *--v26;
        *v26 = *(_QWORD *)a1;
        result = sub_BD9F80((__int64)a1, 0, ((char *)v26 - a1) >> 3, v38);
      }
      while ( (char *)v26 - a1 > 8 );
    }
    return result;
  }
}
