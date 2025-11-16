// Function: sub_18F35E0
// Address: 0x18f35e0
//
void __fastcall sub_18F35E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rdi
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // r15
  __int64 v11; // r13
  int v12; // r12d
  __int64 *v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // r14
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // eax
  __int64 *v23; // rdx
  __int64 v24; // rdi
  unsigned int v25; // eax
  _QWORD *v26; // rdi
  __int64 v27; // rsi
  _QWORD *v28; // rax
  int v29; // r9d
  int v30; // eax
  int v31; // edx
  __int64 v32; // rcx
  unsigned int v33; // esi
  __int64 *v34; // rax
  __int64 v35; // rdi
  int v36; // eax
  __int64 v37; // rsi
  int v38; // edx
  unsigned int v39; // eax
  int v40; // edi
  __int64 *v41; // r12
  __int64 v42; // rcx
  __int64 v43; // r15
  __int64 v44; // rdi
  _QWORD *v45; // rdi
  int v46; // eax
  int v47; // eax
  int v48; // r8d
  int v49; // edx
  int v50; // r8d
  __int64 v54; // [rsp+28h] [rbp-168h]
  __int64 v57; // [rsp+48h] [rbp-148h] BYREF
  _QWORD *v58; // [rsp+50h] [rbp-140h] BYREF
  __int64 v59; // [rsp+58h] [rbp-138h]
  _QWORD v60[38]; // [rsp+60h] [rbp-130h] BYREF

  v60[0] = a1;
  v59 = 0x2000000001LL;
  v7 = *a2;
  v58 = v60;
  v54 = v7;
  v8 = v60;
  v9 = 1;
  do
  {
    v10 = 0;
    v11 = v8[v9 - 1];
    LODWORD(v59) = v9 - 1;
    sub_1AEAA40(v11);
    sub_14191F0(a3, v11);
    if ( (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) != 0 )
    {
      v12 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      do
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
            v13 = (__int64 *)(*(_QWORD *)(v11 - 8) + 24 * v10);
          else
            v13 = (__int64 *)(v11 + 24 * (v10 - (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)));
          v16 = *v13;
          if ( *v13 )
          {
            v14 = v13[1];
            v15 = v13[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v15 = v14;
            if ( v14 )
              *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
          }
          *v13 = 0;
          if ( !*(_QWORD *)(v16 + 8) && *(_BYTE *)(v16 + 16) > 0x17u && (unsigned __int8)sub_1AE9990(v16, a4) )
            break;
          if ( v12 == (_DWORD)++v10 )
            goto LABEL_17;
        }
        v19 = (unsigned int)v59;
        if ( (unsigned int)v59 >= HIDWORD(v59) )
        {
          sub_16CD150((__int64)&v58, v60, 0, 8, v17, v18);
          v19 = (unsigned int)v59;
        }
        ++v10;
        v58[v19] = v16;
        LODWORD(v59) = v59 + 1;
      }
      while ( v12 != (_DWORD)v10 );
    }
LABEL_17:
    if ( a7 )
    {
      v57 = v11;
      if ( (*(_BYTE *)(a7 + 8) & 1) != 0 )
      {
        v20 = a7 + 16;
        v21 = 15;
      }
      else
      {
        v46 = *(_DWORD *)(a7 + 24);
        v20 = *(_QWORD *)(a7 + 16);
        if ( !v46 )
          goto LABEL_24;
        v21 = v46 - 1;
      }
      v22 = v21 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v23 = (__int64 *)(v20 + 8LL * v22);
      v24 = *v23;
      if ( v11 == *v23 )
      {
LABEL_21:
        *v23 = -16;
        v25 = *(_DWORD *)(a7 + 8);
        ++*(_DWORD *)(a7 + 12);
        v26 = *(_QWORD **)(a7 + 144);
        *(_DWORD *)(a7 + 8) = (2 * (v25 >> 1) - 2) | v25 & 1;
        v27 = (__int64)&v26[*(unsigned int *)(a7 + 152)];
        v28 = sub_18F2A10(v26, v27, &v57);
        if ( v28 + 1 != (_QWORD *)v27 )
        {
          memmove(v28, v28 + 1, v27 - (_QWORD)(v28 + 1));
          v29 = *(_DWORD *)(a7 + 152);
        }
        *(_DWORD *)(a7 + 152) = v29 - 1;
      }
      else
      {
        v49 = 1;
        while ( v24 != -8 )
        {
          v50 = v49 + 1;
          v22 = v21 & (v49 + v22);
          v23 = (__int64 *)(v20 + 8LL * v22);
          v24 = *v23;
          if ( v11 == *v23 )
            goto LABEL_21;
          v49 = v50;
        }
      }
    }
LABEL_24:
    v30 = *(_DWORD *)(a6 + 24);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a6 + 8);
      v33 = (v30 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v34 = (__int64 *)(v32 + 16LL * v33);
      v35 = *v34;
      if ( v11 == *v34 )
      {
LABEL_26:
        *v34 = -16;
        --*(_DWORD *)(a6 + 16);
        ++*(_DWORD *)(a6 + 20);
      }
      else
      {
        v47 = 1;
        while ( v35 != -8 )
        {
          v48 = v47 + 1;
          v33 = v31 & (v47 + v33);
          v34 = (__int64 *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v11 == *v34 )
            goto LABEL_26;
          v47 = v48;
        }
      }
    }
    v36 = *(_DWORD *)(a5 + 24);
    if ( !v36 )
      goto LABEL_32;
    v37 = *(_QWORD *)(a5 + 8);
    v38 = v36 - 1;
    v39 = (v36 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v40 = 1;
    v41 = (__int64 *)(v37 + 56LL * v39);
    v42 = *v41;
    if ( *v41 == v11 )
    {
LABEL_29:
      v43 = v41[3];
      while ( v43 )
      {
        sub_18F3410(*(_QWORD *)(v43 + 24));
        v44 = v43;
        v43 = *(_QWORD *)(v43 + 16);
        j_j___libc_free_0(v44, 48);
      }
      *v41 = -16;
      --*(_DWORD *)(a5 + 16);
      ++*(_DWORD *)(a5 + 20);
LABEL_32:
      v45 = (_QWORD *)v11;
      if ( v54 != v11 + 24 )
        goto LABEL_33;
      goto LABEL_42;
    }
    while ( v42 != -8 )
    {
      v39 = v38 & (v40 + v39);
      v41 = (__int64 *)(v37 + 56LL * v39);
      v42 = *v41;
      if ( v11 == *v41 )
        goto LABEL_29;
      ++v40;
    }
    v45 = (_QWORD *)v11;
    if ( v54 != v11 + 24 )
    {
LABEL_33:
      sub_15F20C0(v45);
      goto LABEL_34;
    }
LABEL_42:
    v54 = sub_15F20C0(v45);
LABEL_34:
    v9 = v59;
    v8 = v58;
  }
  while ( (_DWORD)v59 );
  *a2 = v54;
  if ( v8 != v60 )
    _libc_free((unsigned __int64)v8);
}
