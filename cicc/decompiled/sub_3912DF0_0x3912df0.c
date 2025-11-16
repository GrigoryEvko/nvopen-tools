// Function: sub_3912DF0
// Address: 0x3912df0
//
__int64 __fastcall sub_3912DF0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6)
{
  unsigned __int64 v9; // rcx
  unsigned int v10; // r12d
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  _DWORD *v15; // rax
  unsigned int v17; // esi
  int v18; // eax
  __int64 v19; // r12
  unsigned int v20; // ebx
  _DWORD *v21; // rax
  unsigned int v22; // esi
  _DWORD *v23; // r15
  __int64 v24; // r10
  __int64 v25; // rdi
  unsigned int v26; // ecx
  unsigned int *v27; // rax
  unsigned int v28; // edx
  unsigned __int64 v29; // rdi
  __int64 v30; // rbx
  __int64 v31; // r12
  unsigned __int64 v32; // rdi
  int v33; // r11d
  unsigned int *v34; // r8
  int v35; // edx
  int v36; // edx
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // r9
  unsigned int v40; // esi
  unsigned int v41; // edi
  int v42; // r8d
  unsigned int *v43; // r10
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // r9
  int v47; // r8d
  unsigned int v48; // esi
  unsigned int v49; // edi
  unsigned int v50; // [rsp+0h] [rbp-50h]
  unsigned int v51; // [rsp+4h] [rbp-4Ch]
  __int64 v52; // [rsp+8h] [rbp-48h]
  int v53; // [rsp+10h] [rbp-40h]
  unsigned __int64 v54; // [rsp+10h] [rbp-40h]
  unsigned int v55; // [rsp+10h] [rbp-40h]
  unsigned int v56; // [rsp+18h] [rbp-38h]
  __int64 v57; // [rsp+18h] [rbp-38h]
  unsigned __int64 v58; // [rsp+18h] [rbp-38h]

  v9 = a2;
  v10 = a2;
  v11 = a1;
  v12 = *(_QWORD *)(a1 + 296);
  v13 = *(_QWORD *)(a1 + 288);
  v14 = 0x6DB6DB6DB6DB6DB7LL * ((v12 - v13) >> 3);
  if ( v9 >= v14 )
  {
    v29 = v10 + 1;
    if ( v29 > v14 )
    {
      v55 = a6;
      v58 = v9;
      sub_3911920((unsigned __int64 *)(v11 + 288), v29 - v14);
      v13 = *(_QWORD *)(v11 + 288);
      v9 = v58;
      a6 = v55;
    }
    else if ( v29 < v14 )
    {
      v57 = v13 + 56 * v29;
      if ( v12 != v57 )
      {
        v54 = v9;
        v50 = a6;
        v52 = v11;
        v30 = v12;
        v51 = v10;
        v31 = v13 + 56 * v29;
        do
        {
          v32 = *(_QWORD *)(v31 + 32);
          v31 += 56;
          j___libc_free_0(v32);
        }
        while ( v30 != v31 );
        v11 = v52;
        v9 = v54;
        v10 = v51;
        *(_QWORD *)(v52 + 296) = v57;
        a6 = v50;
        v13 = *(_QWORD *)(v52 + 288);
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
    v56 = v10;
    v17 = a3;
    v18 = 37 * v10;
    v19 = v11;
    v20 = a6;
    v53 = v18;
    while ( 1 )
    {
      v21 = sub_390FE40(v19, v17);
      v22 = v21[12];
      v23 = v21;
      v24 = (__int64)(v21 + 6);
      if ( v22 )
      {
        v25 = *((_QWORD *)v21 + 4);
        v26 = (v22 - 1) & v53;
        v27 = (unsigned int *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( *v27 == v56 )
          goto LABEL_8;
        v33 = 1;
        v34 = 0;
        while ( v28 != -1 )
        {
          if ( !v34 && v28 == -2 )
            v34 = v27;
          v26 = (v22 - 1) & (v33 + v26);
          v27 = (unsigned int *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( *v27 == v56 )
            goto LABEL_8;
          ++v33;
        }
        v35 = v23[10];
        if ( v34 )
          v27 = v34;
        ++*((_QWORD *)v23 + 3);
        v36 = v35 + 1;
        if ( 4 * v36 < 3 * v22 )
        {
          if ( v22 - v23[11] - v36 > v22 >> 3 )
            goto LABEL_24;
          sub_3912C30(v24, v22);
          v44 = v23[12];
          if ( !v44 )
          {
LABEL_53:
            ++v23[10];
            BUG();
          }
          v45 = v44 - 1;
          v46 = *((_QWORD *)v23 + 4);
          v47 = 1;
          v43 = 0;
          v48 = v45 & v53;
          v36 = v23[10] + 1;
          v27 = (unsigned int *)(v46 + 16LL * (v45 & (unsigned int)v53));
          v49 = *v27;
          if ( *v27 == v56 )
            goto LABEL_24;
          while ( v49 != -1 )
          {
            if ( v49 == -2 && !v43 )
              v43 = v27;
            v48 = v45 & (v47 + v48);
            v27 = (unsigned int *)(v46 + 16LL * v48);
            v49 = *v27;
            if ( *v27 == v56 )
              goto LABEL_24;
            ++v47;
          }
          goto LABEL_32;
        }
      }
      else
      {
        ++*((_QWORD *)v21 + 3);
      }
      sub_3912C30(v24, 2 * v22);
      v37 = v23[12];
      if ( !v37 )
        goto LABEL_53;
      v38 = v37 - 1;
      v39 = *((_QWORD *)v23 + 4);
      v40 = v38 & v53;
      v36 = v23[10] + 1;
      v27 = (unsigned int *)(v39 + 16LL * (v38 & (unsigned int)v53));
      v41 = *v27;
      if ( *v27 == v56 )
        goto LABEL_24;
      v42 = 1;
      v43 = 0;
      while ( v41 != -1 )
      {
        if ( !v43 && v41 == -2 )
          v43 = v27;
        v40 = v38 & (v42 + v40);
        v27 = (unsigned int *)(v39 + 16LL * v40);
        v41 = *v27;
        if ( *v27 == v56 )
          goto LABEL_24;
        ++v42;
      }
LABEL_32:
      if ( v43 )
        v27 = v43;
LABEL_24:
      v23[10] = v36;
      if ( *v27 != -1 )
        --v23[11];
      *(_QWORD *)(v27 + 1) = 0;
      v27[3] = 0;
      *v27 = v56;
LABEL_8:
      v27[1] = a4;
      v27[2] = a5;
      v27[3] = v20;
      v17 = *v23 - 1;
      if ( v17 > 0xFFFFFFFD )
        return 1;
      a4 = v23[1];
      v20 = v23[3];
      a5 = v23[2];
    }
  }
  return 0;
}
