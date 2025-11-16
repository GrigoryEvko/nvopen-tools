// Function: sub_29ABD40
// Address: 0x29abd40
//
void __fastcall sub_29ABD40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rbx
  __int64 *v13; // r13
  _BYTE *v14; // r15
  int v15; // esi
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // esi
  unsigned int v19; // eax
  __int64 v20; // rcx
  int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // r9
  __int64 v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // r13
  __int64 v30; // r8
  int v31; // ecx
  __int64 v32; // r10
  __int64 *v33; // rax
  __int64 v34; // rdi
  int v35; // ecx
  unsigned int v36; // edx
  __int64 v37; // r11
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // r15
  __int64 v42; // rbx
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // r9
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r12
  __int64 *v54; // r13
  __int64 v55; // r15
  __int64 v56; // rbx
  __int64 *v57; // r14
  __int64 v58; // rdi
  unsigned __int64 v59; // rax
  __int64 v60; // [rsp+0h] [rbp-90h]
  __int64 v65; // [rsp+30h] [rbp-60h]
  __int64 *v66; // [rsp+40h] [rbp-50h]
  __int64 v67; // [rsp+40h] [rbp-50h]
  __int64 *v68; // [rsp+48h] [rbp-48h]
  unsigned __int64 v69; // [rsp+48h] [rbp-48h]
  __int64 v70; // [rsp+58h] [rbp-38h]
  __int64 v71; // [rsp+58h] [rbp-38h]
  __int64 v72; // [rsp+58h] [rbp-38h]
  int v73; // [rsp+58h] [rbp-38h]

  v7 = a1;
  v8 = *(_QWORD *)(a3 + 16);
  if ( v8 )
  {
    v10 = *(_QWORD *)(a3 + 16);
    v11 = 0;
    do
    {
      v10 = *(_QWORD *)(v10 + 8);
      ++v11;
    }
    while ( v10 );
    if ( v11 > 0xFFFFFFFFFFFFFFFLL )
LABEL_61:
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v65 = sub_22077B0(8 * v11);
    v12 = (__int64 *)v65;
    do
    {
      *v12++ = *(_QWORD *)(v8 + 24);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v8 );
    if ( v12 != (__int64 *)v65 )
    {
      v70 = a3;
      v13 = (__int64 *)v65;
      while ( 1 )
      {
        v14 = (_BYTE *)*v13;
        if ( (unsigned __int8)(*(_BYTE *)*v13 - 30) > 0xAu || a2 != sub_B43CB0(*v13) )
          goto LABEL_9;
        v15 = *(_DWORD *)(a1 + 80);
        v16 = *((_QWORD *)v14 + 5);
        v17 = *(_QWORD *)(a1 + 64);
        if ( !v15 )
          goto LABEL_60;
        v18 = v15 - 1;
        v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v20 = *(_QWORD *)(v17 + 8LL * v19);
        if ( v16 != v20 )
          break;
LABEL_9:
        if ( v12 == ++v13 )
          goto LABEL_19;
      }
      v21 = 1;
      while ( v20 != -4096 )
      {
        v19 = v18 & (v21 + v19);
        v20 = *(_QWORD *)(v17 + 8LL * v19);
        if ( v16 == v20 )
          goto LABEL_9;
        ++v21;
      }
LABEL_60:
      sub_BD2ED0((__int64)v14, v70, a4);
      goto LABEL_9;
    }
  }
  else
  {
    v65 = 0;
  }
LABEL_19:
  v22 = *(__int64 **)(a1 + 104);
  v68 = v22;
  v66 = &v22[*(unsigned int *)(a1 + 112)];
  if ( v22 != v66 )
  {
    while ( 1 )
    {
      v23 = sub_AA5930(*v68);
      v25 = v24;
      v26 = v23;
LABEL_21:
      if ( v25 != v26 )
        break;
LABEL_32:
      if ( v66 == ++v68 )
        goto LABEL_33;
    }
    while ( (*(_DWORD *)(v26 + 4) & 0x7FFFFFF) == 0 )
    {
LABEL_29:
      v38 = *(_QWORD *)(v26 + 32);
      if ( !v38 )
        BUG();
      v26 = 0;
      if ( *(_BYTE *)(v38 - 24) != 84 )
        goto LABEL_21;
      v26 = v38 - 24;
      if ( v25 == v38 - 24 )
        goto LABEL_32;
    }
    v27 = *(_QWORD *)(v26 - 8);
    v28 = 0;
    v29 = 0;
    v30 = 8LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
    while ( 1 )
    {
      v31 = *(_DWORD *)(v7 + 80);
      v32 = *(_QWORD *)(v7 + 64);
      v33 = (__int64 *)(v27 + v28 + 32LL * *(unsigned int *)(v26 + 72));
      v34 = *v33;
      if ( !v31 )
        goto LABEL_24;
      v35 = v31 - 1;
      v36 = v35 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v37 = *(_QWORD *)(v32 + 8LL * v36);
      if ( v34 != v37 )
      {
        v73 = 1;
        while ( v37 != -4096 )
        {
          v36 = v35 & (v73 + v36);
          ++v73;
          v37 = *(_QWORD *)(v32 + 8LL * v36);
          if ( v34 == v37 )
            goto LABEL_27;
        }
        goto LABEL_24;
      }
LABEL_27:
      if ( v29 )
      {
LABEL_24:
        v28 += 8;
        if ( v30 == v28 )
          goto LABEL_29;
      }
      else
      {
        *v33 = a4;
        v27 = *(_QWORD *)(v26 - 8);
        v29 = *(_QWORD *)(v27 + 4 * v28);
        v28 += 8;
        if ( v30 == v28 )
          goto LABEL_29;
      }
    }
  }
LABEL_33:
  v39 = *(unsigned int *)(a5 + 40);
  if ( (_DWORD)v39 )
  {
    v60 = v7;
    v40 = a5;
    v67 = 8 * v39;
    v41 = 0;
    v42 = a2;
    do
    {
      v43 = *(_QWORD *)(a7 + v41);
      v44 = *(_QWORD *)(*(_QWORD *)(v40 + 32) + v41);
      v45 = *(_QWORD *)(v44 + 16);
      if ( v45 )
      {
        v46 = *(_QWORD *)(v44 + 16);
        v47 = 0;
        do
        {
          v46 = *(_QWORD *)(v46 + 8);
          ++v47;
        }
        while ( v46 );
        if ( v47 > 0xFFFFFFFFFFFFFFFLL )
          goto LABEL_61;
        v71 = v45;
        v48 = sub_22077B0(8 * v47);
        v49 = v71;
        v50 = v48;
        v51 = v48;
        do
        {
          v51 += 8;
          *(_QWORD *)(v51 - 8) = *(_QWORD *)(v49 + 24);
          v49 = *(_QWORD *)(v49 + 8);
        }
        while ( v49 );
        if ( v51 == v48 )
          goto LABEL_47;
        v52 = v40;
        v53 = v41;
        v54 = (__int64 *)v51;
        v55 = v42;
        v69 = v48;
        v56 = v43;
        v57 = (__int64 *)v48;
        do
        {
          while ( 1 )
          {
            v58 = *v57;
            if ( v55 == *(_QWORD *)(*(_QWORD *)(*v57 + 40) + 72LL) )
              break;
            if ( v54 == ++v57 )
              goto LABEL_46;
          }
          ++v57;
          v72 = v52;
          sub_BD2ED0(v58, *(_QWORD *)(*(_QWORD *)(v52 + 32) + v53), v56);
          v52 = v72;
        }
        while ( v54 != v57 );
LABEL_46:
        v50 = v69;
        v42 = v55;
        v40 = v52;
        v41 = v53;
        if ( v69 )
LABEL_47:
          j_j___libc_free_0(v50);
      }
      v41 += 8;
    }
    while ( v67 != v41 );
    v7 = v60;
  }
  if ( *(_QWORD *)(v7 + 16) && *(_DWORD *)(v7 + 112) > 1u )
  {
    sub_29AB5B0(v7, a4, a6, *(_QWORD *)(v7 + 24));
    v59 = v65;
    if ( v65 )
      goto LABEL_53;
  }
  else
  {
    v59 = v65;
    if ( v65 )
LABEL_53:
      j_j___libc_free_0(v59);
  }
}
