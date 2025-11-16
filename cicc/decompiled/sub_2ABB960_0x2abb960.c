// Function: sub_2ABB960
// Address: 0x2abb960
//
bool __fastcall sub_2ABB960(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 *v8; // r14
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rsi
  int v12; // r10d
  int v13; // edx
  unsigned int v14; // eax
  int *v15; // r9
  int v16; // r8d
  unsigned int v17; // eax
  bool result; // al
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // rsi
  int v24; // r10d
  int v25; // edx
  unsigned int v26; // eax
  int *v27; // r9
  int v28; // r8d
  unsigned int v29; // eax
  __int64 v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rsi
  int v35; // r10d
  int v36; // edx
  unsigned int v37; // eax
  int *v38; // r9
  int v39; // r8d
  unsigned int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rsi
  __int64 v43; // rax
  int v44; // edx
  __int64 v45; // rsi
  int v46; // r10d
  int v47; // edx
  unsigned int v48; // eax
  int *v49; // r9
  int v50; // r8d
  unsigned int v51; // eax
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 *v54; // [rsp+8h] [rbp-48h]
  _QWORD v55[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(__int64 **)a1;
  v4 = 8LL * *(unsigned int *)(a1 + 8);
  v54 = (__int64 *)(*(_QWORD *)a1 + v4);
  v5 = v4 >> 3;
  v6 = v4 >> 5;
  if ( v6 )
  {
    v7 = *a2;
    v8 = &v3[4 * v6];
    while ( 1 )
    {
      v9 = *v3;
      v55[0] = v7;
      if ( *(_DWORD *)(v9 + 64) )
        break;
      v19 = *(_QWORD *)(v9 + 80);
      v20 = v19 + 8LL * *(unsigned int *)(v9 + 88);
      if ( v20 != sub_2AA8390(v19, v20, (int *)v55) )
        return v54 != v3;
LABEL_12:
      v21 = v3[1];
      v55[0] = v7;
      if ( *(_DWORD *)(v21 + 64) )
      {
        v22 = *(_DWORD *)(v21 + 72);
        v23 = *(_QWORD *)(v21 + 56);
        if ( !v22 )
          goto LABEL_20;
        v24 = 1;
        v25 = v22 - 1;
        v26 = v25 & ((BYTE4(v55[0]) == 0) + 37 * LODWORD(v55[0]) - 1);
        v27 = (int *)(v23 + 8LL * v26);
        v28 = *v27;
        if ( LODWORD(v55[0]) == *v27 )
          goto LABEL_17;
        while ( 1 )
        {
          do
          {
            if ( v28 == -1 && *((_BYTE *)v27 + 4) )
              goto LABEL_20;
            v29 = v24 + v26;
            ++v24;
            v26 = v25 & v29;
            v27 = (int *)(v23 + 8LL * v26);
            v28 = *v27;
          }
          while ( LODWORD(v55[0]) != *v27 );
LABEL_17:
          if ( BYTE4(v55[0]) == *((_BYTE *)v27 + 4) )
            return v54 != v3 + 1;
        }
      }
      v30 = *(_QWORD *)(v21 + 80);
      v31 = v30 + 8LL * *(unsigned int *)(v21 + 88);
      if ( v31 != sub_2AA8390(v30, v31, (int *)v55) )
        return v54 != v3 + 1;
LABEL_20:
      v32 = v3[2];
      v55[0] = v7;
      if ( *(_DWORD *)(v32 + 64) )
      {
        v33 = *(_DWORD *)(v32 + 72);
        v34 = *(_QWORD *)(v32 + 56);
        if ( !v33 )
          goto LABEL_28;
        v35 = 1;
        v36 = v33 - 1;
        v37 = v36 & ((BYTE4(v55[0]) == 0) + 37 * LODWORD(v55[0]) - 1);
        v38 = (int *)(v34 + 8LL * v37);
        v39 = *v38;
        if ( LODWORD(v55[0]) == *v38 )
          goto LABEL_25;
        while ( 1 )
        {
          do
          {
            if ( v39 == -1 && *((_BYTE *)v38 + 4) )
              goto LABEL_28;
            v40 = v35 + v37;
            ++v35;
            v37 = v36 & v40;
            v38 = (int *)(v34 + 8LL * v37);
            v39 = *v38;
          }
          while ( LODWORD(v55[0]) != *v38 );
LABEL_25:
          if ( BYTE4(v55[0]) == *((_BYTE *)v38 + 4) )
            return v54 != v3 + 2;
        }
      }
      v41 = *(_QWORD *)(v32 + 80);
      v42 = v41 + 8LL * *(unsigned int *)(v32 + 88);
      if ( v42 != sub_2AA8390(v41, v42, (int *)v55) )
        return v54 != v3 + 2;
LABEL_28:
      v43 = v3[3];
      v55[0] = v7;
      if ( *(_DWORD *)(v43 + 64) )
      {
        v44 = *(_DWORD *)(v43 + 72);
        v45 = *(_QWORD *)(v43 + 56);
        if ( v44 )
        {
          v46 = 1;
          v47 = v44 - 1;
          v48 = v47 & ((BYTE4(v55[0]) == 0) + 37 * LODWORD(v55[0]) - 1);
          v49 = (int *)(v45 + 8LL * v48);
          v50 = *v49;
          if ( LODWORD(v55[0]) == *v49 )
            goto LABEL_33;
          while ( 1 )
          {
            do
            {
              if ( v50 == -1 && *((_BYTE *)v49 + 4) )
                goto LABEL_36;
              v51 = v46 + v48;
              ++v46;
              v48 = v47 & v51;
              v49 = (int *)(v45 + 8LL * v48);
              v50 = *v49;
            }
            while ( LODWORD(v55[0]) != *v49 );
LABEL_33:
            if ( BYTE4(v55[0]) == *((_BYTE *)v49 + 4) )
              return v54 != v3 + 3;
          }
        }
      }
      else
      {
        v52 = *(_QWORD *)(v43 + 80);
        v53 = v52 + 8LL * *(unsigned int *)(v43 + 88);
        if ( v53 != sub_2AA8390(v52, v53, (int *)v55) )
          return v54 != v3 + 3;
      }
LABEL_36:
      v3 += 4;
      if ( v8 == v3 )
      {
        v5 = v54 - v3;
        goto LABEL_38;
      }
    }
    v10 = *(_DWORD *)(v9 + 72);
    v11 = *(_QWORD *)(v9 + 56);
    if ( !v10 )
      goto LABEL_12;
    v12 = 1;
    v13 = v10 - 1;
    v14 = v13 & ((BYTE4(v55[0]) == 0) + 37 * LODWORD(v55[0]) - 1);
    v15 = (int *)(v11 + 8LL * v14);
    v16 = *v15;
    if ( LODWORD(v55[0]) == *v15 )
      goto LABEL_8;
    while ( 1 )
    {
      do
      {
        if ( v16 == -1 && *((_BYTE *)v15 + 4) )
          goto LABEL_12;
        v17 = v12 + v14;
        ++v12;
        v14 = v13 & v17;
        v15 = (int *)(v11 + 8LL * v14);
        v16 = *v15;
      }
      while ( LODWORD(v55[0]) != *v15 );
LABEL_8:
      if ( BYTE4(v55[0]) == *((_BYTE *)v15 + 4) )
        return v54 != v3;
    }
  }
LABEL_38:
  if ( v5 == 2 )
    goto LABEL_44;
  if ( v5 == 3 )
  {
    if ( sub_2AA90B0(a2, v3) )
      return v54 != v3;
    ++v3;
LABEL_44:
    if ( !sub_2AA90B0(a2, v3) )
    {
      ++v3;
      goto LABEL_46;
    }
    return v54 != v3;
  }
  if ( v5 != 1 )
    return 0;
LABEL_46:
  result = sub_2AA90B0(a2, v3);
  if ( result )
    return v54 != v3;
  return result;
}
