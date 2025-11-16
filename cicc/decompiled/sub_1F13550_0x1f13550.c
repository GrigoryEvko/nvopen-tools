// Function: sub_1F13550
// Address: 0x1f13550
//
__int64 __fastcall sub_1F13550(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  int v6; // eax
  int v7; // edi
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r9
  int v20; // eax
  __int64 v21; // rcx
  int v22; // edx
  unsigned int v23; // edi
  __int64 *v24; // rax
  __int64 v25; // r8
  _QWORD **v26; // rcx
  _QWORD *v27; // rax
  unsigned int i; // edx
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // ecx
  __int64 *v35; // rdx
  __int64 v36; // r9
  __int64 *v37; // r15
  __int64 v38; // r14
  __int64 v39; // rdi
  unsigned int v40; // edx
  __int64 *v41; // rdx
  __int64 *v42; // rcx
  int v43; // eax
  int v45; // edx
  int v46; // r10d
  int v47; // r9d
  __int64 v48; // rax
  int v49; // eax
  int v50; // edx
  int v51; // r10d
  int v52; // r9d
  _QWORD **v53; // [rsp+8h] [rbp-48h]
  __int64 v54; // [rsp+10h] [rbp-40h]
  unsigned int v55; // [rsp+18h] [rbp-38h]

  v53 = 0;
  v5 = *(_QWORD *)(*a1 + 24LL);
  v6 = *(_DWORD *)(v5 + 256);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = *(_QWORD *)(v5 + 240);
    v9 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a3 == *v10 )
    {
LABEL_3:
      v53 = (_QWORD **)v10[1];
    }
    else
    {
      v49 = 1;
      while ( v11 != -8 )
      {
        v52 = v49 + 1;
        v9 = v7 & (v49 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( a3 == *v10 )
          goto LABEL_3;
        v49 = v52;
      }
      v53 = 0;
    }
  }
  v12 = a1[5];
  sub_1E06620(v12);
  v13 = *(_QWORD *)(v12 + 1312);
  v14 = 0;
  v15 = *(unsigned int *)(v13 + 48);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD *)(v13 + 32);
    v17 = (v15 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v18 = (__int64 *)(v16 + 16LL * v17);
    v19 = *v18;
    if ( a3 == *v18 )
    {
LABEL_6:
      if ( v18 != (__int64 *)(v16 + 16 * v15) )
      {
        v14 = v18[1];
        goto LABEL_8;
      }
    }
    else
    {
      v50 = 1;
      while ( v19 != -8 )
      {
        v51 = v50 + 1;
        v17 = (v15 - 1) & (v50 + v17);
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( a3 == *v18 )
          goto LABEL_6;
        v50 = v51;
      }
    }
    v14 = 0;
  }
LABEL_8:
  v20 = *(_DWORD *)(v5 + 256);
  if ( v20 )
  {
    v54 = a2;
    v21 = *(_QWORD *)(v5 + 240);
    v55 = -1;
    while ( 1 )
    {
      v22 = v20 - 1;
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( *v24 != a2 )
      {
        v43 = 1;
        while ( v25 != -8 )
        {
          v47 = v43 + 1;
          v48 = v22 & (v23 + v43);
          v23 = v48;
          v24 = (__int64 *)(v21 + 16 * v48);
          v25 = *v24;
          if ( *v24 == a2 )
            goto LABEL_11;
          v43 = v47;
        }
        return a2;
      }
LABEL_11:
      v26 = (_QWORD **)v24[1];
      if ( !v26 || v53 == v26 )
        return a2;
      v27 = *v26;
      for ( i = 1; v27; ++i )
        v27 = (_QWORD *)*v27;
      if ( i < v55 )
      {
        v55 = i;
        v54 = a2;
      }
      v29 = a1[5];
      v30 = *v26[4];
      sub_1E06620(v29);
      v31 = *(_QWORD *)(v29 + 1312);
      v32 = *(unsigned int *)(v31 + 48);
      if ( !(_DWORD)v32 )
        goto LABEL_58;
      v33 = *(_QWORD *)(v31 + 32);
      v34 = (v32 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v35 = (__int64 *)(v33 + 16LL * v34);
      v36 = *v35;
      if ( v30 != *v35 )
      {
        v45 = 1;
        while ( v36 != -8 )
        {
          v46 = v45 + 1;
          v34 = (v32 - 1) & (v45 + v34);
          v35 = (__int64 *)(v33 + 16LL * v34);
          v36 = *v35;
          if ( v30 == *v35 )
            goto LABEL_19;
          v45 = v46;
        }
LABEL_58:
        BUG();
      }
LABEL_19:
      if ( v35 == (__int64 *)(v33 + 16 * v32) )
        goto LABEL_58;
      v37 = *(__int64 **)(v35[1] + 8);
      if ( !v37 )
        return v54;
      v38 = a1[5];
      sub_1E06620(v38);
      if ( v37 != (__int64 *)v14 )
      {
        if ( !v14 )
          return v54;
        if ( v14 != v37[1] )
        {
          if ( v37 == *(__int64 **)(v14 + 8) || *(_DWORD *)(v14 + 16) >= *((_DWORD *)v37 + 4) )
            return v54;
          v39 = *(_QWORD *)(v38 + 1312);
          if ( *(_BYTE *)(v39 + 72) )
            goto LABEL_39;
          v40 = *(_DWORD *)(v39 + 76) + 1;
          *(_DWORD *)(v39 + 76) = v40;
          if ( v40 > 0x20 )
          {
            sub_1E052A0(v39);
LABEL_39:
            if ( *((_DWORD *)v37 + 12) < *(_DWORD *)(v14 + 48) || *((_DWORD *)v37 + 13) > *(_DWORD *)(v14 + 52) )
              return v54;
            goto LABEL_32;
          }
          v41 = v37;
          do
          {
            v42 = v41;
            v41 = (__int64 *)v41[1];
          }
          while ( v41 && *(_DWORD *)(v14 + 16) <= *((_DWORD *)v41 + 4) );
          if ( (__int64 *)v14 != v42 )
            return v54;
        }
      }
LABEL_32:
      v20 = *(_DWORD *)(v5 + 256);
      a2 = *v37;
      if ( !v20 )
        return a2;
      v21 = *(_QWORD *)(v5 + 240);
    }
  }
  return a2;
}
