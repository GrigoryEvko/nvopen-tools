// Function: sub_E626D0
// Address: 0xe626d0
//
__int64 __fastcall sub_E626D0(__int64 a1, unsigned int a2, unsigned int a3, int a4, int a5, int a6)
{
  unsigned __int64 v9; // rcx
  unsigned int v10; // r12d
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  _DWORD *v15; // rdx
  unsigned int v17; // esi
  int v18; // eax
  __int64 v19; // r12
  int v20; // ebx
  _DWORD *v21; // rax
  unsigned int v22; // esi
  _DWORD *v23; // r15
  __int64 v24; // r10
  __int64 v25; // r9
  unsigned int *v26; // rdx
  int v27; // r8d
  unsigned int v28; // edi
  unsigned int *v29; // rax
  unsigned int v30; // ecx
  _DWORD *v31; // rax
  unsigned __int64 v32; // rdi
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rdi
  int v37; // eax
  int v38; // eax
  int v39; // ecx
  int v40; // ecx
  __int64 v41; // r9
  unsigned int v42; // esi
  unsigned int v43; // edi
  int v44; // r8d
  unsigned int *v45; // r10
  int v46; // ecx
  int v47; // ecx
  __int64 v48; // r9
  int v49; // r8d
  unsigned int v50; // esi
  unsigned int v51; // edi
  int v52; // [rsp+0h] [rbp-50h]
  unsigned int v53; // [rsp+4h] [rbp-4Ch]
  __int64 v54; // [rsp+8h] [rbp-48h]
  int v55; // [rsp+10h] [rbp-40h]
  unsigned __int64 v56; // [rsp+10h] [rbp-40h]
  int v57; // [rsp+10h] [rbp-40h]
  unsigned int v58; // [rsp+18h] [rbp-38h]
  __int64 v59; // [rsp+18h] [rbp-38h]
  unsigned __int64 v60; // [rsp+18h] [rbp-38h]

  v9 = a2;
  v10 = a2;
  v11 = a1;
  v12 = *(_QWORD *)(a1 + 288);
  v13 = *(_QWORD *)(a1 + 280);
  v14 = 0x6DB6DB6DB6DB6DB7LL * ((v12 - v13) >> 3);
  if ( v9 >= v14 )
  {
    v32 = v10 + 1;
    if ( v32 > v14 )
    {
      v57 = a6;
      v60 = v9;
      sub_E60D70((__int64 *)(v11 + 280), v32 - v14);
      v13 = *(_QWORD *)(v11 + 280);
      v9 = v60;
      a6 = v57;
    }
    else if ( v32 < v14 )
    {
      v59 = v13 + 56 * v32;
      if ( v12 != v59 )
      {
        v56 = v9;
        v52 = a6;
        v54 = v11;
        v33 = v12;
        v53 = v10;
        v34 = v13 + 56 * v32;
        do
        {
          v35 = *(unsigned int *)(v34 + 48);
          v36 = *(_QWORD *)(v34 + 32);
          v34 += 56;
          sub_C7D6A0(v36, 16 * v35, 4);
        }
        while ( v33 != v34 );
        v11 = v54;
        v9 = v56;
        v10 = v53;
        *(_QWORD *)(v54 + 288) = v59;
        a6 = v52;
        v13 = *(_QWORD *)(v54 + 280);
      }
    }
  }
  v15 = (_DWORD *)(v13 + 56 * v9);
  if ( !*v15 )
  {
    v15[1] = a4;
    *v15 = a3 + 1;
    v15[2] = a5;
    v15[3] = a6;
    if ( a3 > 0xFFFFFFFD )
      return 1;
    v58 = v10;
    v17 = a3;
    v18 = 37 * v10;
    v19 = v11;
    v20 = a6;
    v55 = v18;
    while ( 1 )
    {
      v21 = sub_E5F790(v19, v17);
      v22 = v21[12];
      v23 = v21;
      v24 = (__int64)(v21 + 6);
      if ( v22 )
      {
        v25 = *((_QWORD *)v21 + 4);
        v26 = 0;
        v27 = 1;
        v28 = (v22 - 1) & v55;
        v29 = (unsigned int *)(v25 + 16LL * v28);
        v30 = *v29;
        if ( v58 == *v29 )
        {
LABEL_8:
          v31 = v29 + 1;
          goto LABEL_9;
        }
        while ( v30 != -1 )
        {
          if ( !v26 && v30 == -2 )
            v26 = v29;
          v28 = (v22 - 1) & (v27 + v28);
          v29 = (unsigned int *)(v25 + 16LL * v28);
          v30 = *v29;
          if ( *v29 == v58 )
            goto LABEL_8;
          ++v27;
        }
        if ( !v26 )
          v26 = v29;
        v37 = v23[10];
        ++*((_QWORD *)v23 + 3);
        v38 = v37 + 1;
        if ( 4 * v38 < 3 * v22 )
        {
          if ( v22 - v23[11] - v38 > v22 >> 3 )
            goto LABEL_29;
          sub_E624F0(v24, v22);
          v46 = v23[12];
          if ( !v46 )
          {
LABEL_53:
            ++v23[10];
            BUG();
          }
          v47 = v46 - 1;
          v48 = *((_QWORD *)v23 + 4);
          v49 = 1;
          v45 = 0;
          v50 = v47 & v55;
          v38 = v23[10] + 1;
          v26 = (unsigned int *)(v48 + 16LL * (v47 & (unsigned int)v55));
          v51 = *v26;
          if ( *v26 == v58 )
            goto LABEL_29;
          while ( v51 != -1 )
          {
            if ( v51 == -2 && !v45 )
              v45 = v26;
            v50 = v47 & (v49 + v50);
            v26 = (unsigned int *)(v48 + 16LL * v50);
            v51 = *v26;
            if ( *v26 == v58 )
              goto LABEL_29;
            ++v49;
          }
          goto LABEL_37;
        }
      }
      else
      {
        ++*((_QWORD *)v21 + 3);
      }
      sub_E624F0(v24, 2 * v22);
      v39 = v23[12];
      if ( !v39 )
        goto LABEL_53;
      v40 = v39 - 1;
      v41 = *((_QWORD *)v23 + 4);
      v42 = v40 & v55;
      v38 = v23[10] + 1;
      v26 = (unsigned int *)(v41 + 16LL * (v40 & (unsigned int)v55));
      v43 = *v26;
      if ( *v26 == v58 )
        goto LABEL_29;
      v44 = 1;
      v45 = 0;
      while ( v43 != -1 )
      {
        if ( !v45 && v43 == -2 )
          v45 = v26;
        v42 = v40 & (v44 + v42);
        v26 = (unsigned int *)(v41 + 16LL * v42);
        v43 = *v26;
        if ( *v26 == v58 )
          goto LABEL_29;
        ++v44;
      }
LABEL_37:
      if ( v45 )
        v26 = v45;
LABEL_29:
      v23[10] = v38;
      if ( *v26 != -1 )
        --v23[11];
      *(_QWORD *)(v26 + 1) = 0;
      v26[3] = 0;
      *v26 = v58;
      v31 = v26 + 1;
LABEL_9:
      *v31 = a4;
      v31[1] = a5;
      v31[2] = v20;
      v17 = *v23 - 1;
      if ( v17 > 0xFFFFFFFD )
        return 1;
      a5 = v23[2];
      a4 = v23[1];
      v20 = v23[3];
    }
  }
  return 0;
}
