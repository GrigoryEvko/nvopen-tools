// Function: sub_22F4690
// Address: 0x22f4690
//
__int64 *__fastcall sub_22F4690(
        __int64 *a1,
        __int64 *a2,
        __int64 (__fastcall ***a3)(_QWORD),
        int a4,
        char *a5,
        char a6,
        unsigned int *a7)
{
  __int64 v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  int v33; // eax
  int *v34; // rax
  __int64 v35; // r8
  int v36; // esi
  const char *v37; // rdx
  size_t v38; // r9
  size_t v39; // rcx
  size_t v40; // rax
  size_t v41; // rax
  size_t v42; // rax
  __int64 (__fastcall **v43)(_QWORD); // rax
  __int64 (__fastcall *v44)(_QWORD); // rax
  __int64 v45; // rax
  const char *v46; // r10
  int v47; // r8d
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // r15
  __int64 v53; // r14
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  int v57; // ebx
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // rdx
  const char *v62; // rdi
  unsigned int v63; // eax
  __int64 v64; // r14
  __int64 v65; // rax
  const char *v66; // r8
  unsigned int v67; // r14d
  __int64 v68; // rdx
  unsigned __int64 v69; // r9
  unsigned __int64 v70; // r13
  __int64 v71; // rax
  size_t v72; // [rsp+8h] [rbp-1B8h]
  const char *v73; // [rsp+10h] [rbp-1B0h]
  int v74; // [rsp+10h] [rbp-1B0h]
  __int64 v75; // [rsp+18h] [rbp-1A8h]
  int v76; // [rsp+18h] [rbp-1A8h]
  size_t v78; // [rsp+18h] [rbp-1A8h]
  const char *v79; // [rsp+20h] [rbp-1A0h]
  int v80; // [rsp+20h] [rbp-1A0h]
  __int64 v81; // [rsp+20h] [rbp-1A0h]
  size_t v82; // [rsp+20h] [rbp-1A0h]
  __int64 v83; // [rsp+28h] [rbp-198h]
  __int64 v84; // [rsp+28h] [rbp-198h]
  __int64 v85; // [rsp+28h] [rbp-198h]
  __int64 v86; // [rsp+28h] [rbp-198h]
  __int64 v87; // [rsp+28h] [rbp-198h]
  __int64 (__fastcall *v88)(__int64 (__fastcall ***)(_QWORD), _BYTE *, __int64); // [rsp+28h] [rbp-198h]
  const char *v89; // [rsp+28h] [rbp-198h]
  int v90; // [rsp+28h] [rbp-198h]
  int v91; // [rsp+28h] [rbp-198h]
  unsigned int v92; // [rsp+28h] [rbp-198h]
  const char *v93; // [rsp+28h] [rbp-198h]
  const char *v94; // [rsp+28h] [rbp-198h]
  __int64 v95; // [rsp+38h] [rbp-188h] BYREF
  _QWORD v96[4]; // [rsp+40h] [rbp-180h] BYREF
  __int16 v97; // [rsp+60h] [rbp-160h]
  _BYTE *v98; // [rsp+70h] [rbp-150h] BYREF
  __int64 v99; // [rsp+78h] [rbp-148h]
  __int64 v100; // [rsp+80h] [rbp-140h]
  _BYTE v101[312]; // [rsp+88h] [rbp-138h] BYREF

  if ( a6 && (v10 = *a2, *(_BYTE *)(v10 + 44) == 3) )
  {
    v74 = (int)a5;
    v64 = a2[1];
    v92 = *a7;
    v65 = sub_22077B0(0x58u);
    v11 = v65;
    if ( v65 )
      sub_314D310(v65, v10, v64, a4, v74, v92, 0);
    v95 = v11;
  }
  else
  {
    sub_22F3EE0(&v95, a2, a3, a4, a5, a7);
    v11 = v95;
  }
  if ( !v11 )
  {
    *a1 = 0;
    return a1;
  }
  v12 = sub_22F59B0(a2[1], *(unsigned __int16 *)(*a2 + 58));
  v14 = v12;
  if ( !v12 )
    goto LABEL_40;
  v15 = v13;
  v16 = sub_22F59B0(v13, *(unsigned __int16 *)(v12 + 58));
  if ( !v16 )
    goto LABEL_15;
  v83 = v16;
  v15 = v17;
  v18 = sub_22F59B0(v17, *(unsigned __int16 *)(v16 + 58));
  v20 = v83;
  v14 = v18;
  if ( !v18 )
    goto LABEL_59;
  v15 = v19;
  v21 = sub_22F59B0(v19, *(unsigned __int16 *)(v18 + 58));
  if ( !v21 )
    goto LABEL_15;
  v84 = v21;
  v15 = v22;
  v23 = sub_22F59B0(v22, *(unsigned __int16 *)(v21 + 58));
  v20 = v84;
  v14 = v23;
  if ( !v23 )
    goto LABEL_59;
  v15 = v24;
  v25 = sub_22F59B0(v24, *(unsigned __int16 *)(v23 + 58));
  if ( !v25 )
    goto LABEL_15;
  v85 = v25;
  v15 = v26;
  v27 = sub_22F59B0(v26, *(unsigned __int16 *)(v25 + 58));
  v20 = v85;
  v14 = v27;
  if ( v27 )
  {
    v15 = v28;
    v29 = sub_22F59B0(v28, *(unsigned __int16 *)(v27 + 58));
    if ( v29 )
    {
      v86 = v29;
      v15 = v30;
      v98 = (_BYTE *)sub_22F59B0(v30, *(unsigned __int16 *)(v29 + 58));
      v99 = v31;
      v14 = v86;
      if ( v98 )
      {
        v14 = sub_22F33D0(&v98);
        v15 = v32;
      }
    }
  }
  else
  {
LABEL_59:
    v14 = v20;
  }
LABEL_15:
  v33 = *(_DWORD *)(v14 + 40);
  if ( v33 == *(_DWORD *)(*a2 + 40) )
  {
LABEL_40:
    *a1 = v95;
    return a1;
  }
  v34 = (int *)(*(_QWORD *)(v15 + 32) + 80LL * (unsigned int)(v33 - 1));
  v35 = **(_QWORD **)(v15 + 8);
  v87 = *(_QWORD *)(v15 + 16);
  v36 = *v34;
  v37 = (const char *)(v35 + (unsigned int)v34[1]);
  if ( !*v34 )
  {
    v38 = 0;
    v39 = 0;
    v40 = 0;
    if ( !v37 )
      goto LABEL_22;
    goto LABEL_18;
  }
  v62 = (const char *)(v35 + *(unsigned int *)(*(_QWORD *)(v15 + 16) + 4LL * (unsigned int)(v36 + 1)));
  if ( !v62 )
  {
    v38 = 0;
    v41 = 0;
    if ( !v37 )
      goto LABEL_20;
    goto LABEL_18;
  }
  v73 = (const char *)(v35 + (unsigned int)v34[1]);
  v81 = **(_QWORD **)(v15 + 8);
  v63 = strlen(v62);
  v37 = v73;
  v35 = v81;
  v38 = v63;
  v41 = 0;
  if ( v73 )
  {
LABEL_18:
    v72 = v38;
    v75 = v35;
    v79 = v37;
    v41 = strlen(v37);
    v38 = v72;
    v35 = v75;
    v37 = v79;
  }
  if ( v38 > v41 )
  {
    v38 = (size_t)&v37[v41];
    v39 = 0;
    goto LABEL_21;
  }
LABEL_20:
  v42 = v41 - v38;
  v38 += (size_t)v37;
  v39 = v42;
LABEL_21:
  v40 = 0;
  v37 = 0;
  if ( v36 )
  {
    v37 = (const char *)(v35 + *(unsigned int *)(v87 + 4LL * (unsigned int)(v36 + 1)));
    if ( v37 )
    {
      v78 = v39;
      v82 = v38;
      v93 = (const char *)(v35 + *(unsigned int *)(v87 + 4LL * (unsigned int)(v36 + 1)));
      v40 = strlen(v37);
      v39 = v78;
      v38 = v82;
      v37 = v93;
    }
  }
LABEL_22:
  v96[3] = v39;
  v96[1] = v40;
  v43 = *a3;
  v97 = 1285;
  v98 = v101;
  v96[2] = v38;
  v99 = 0;
  v100 = 256;
  v44 = v43[2];
  v96[0] = v37;
  v88 = (__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _BYTE *, __int64))v44;
  sub_CA0EC0((__int64)v96, (__int64)&v98);
  v45 = v88(a3, v98, v99);
  v46 = (const char *)v45;
  if ( v98 != v101 )
  {
    v89 = (const char *)v45;
    _libc_free((unsigned __int64)v98);
    v46 = v89;
  }
  v47 = 0;
  if ( v46 )
  {
    v90 = (int)v46;
    v48 = strlen(v46);
    LODWORD(v46) = v90;
    v47 = v48;
  }
  v76 = (int)v46;
  v80 = v47;
  v91 = *(_DWORD *)(v95 + 40);
  v49 = sub_22077B0(0x58u);
  v52 = v49;
  if ( v49 )
    sub_314D310(v49, v14, v15, v76, v80, v91, 0);
  v53 = v95;
  v54 = *(_QWORD *)(v52 + 80);
  v95 = 0;
  *(_QWORD *)(v52 + 80) = v53;
  if ( v54 )
  {
    sub_314D410(v54);
    j_j___libc_free_0(v54);
  }
  if ( *(_BYTE *)(*a2 + 44) != 3 )
  {
    if ( v53 != v52 )
    {
      v55 = *(unsigned int *)(v53 + 56);
      v56 = *(unsigned int *)(v52 + 56);
      v57 = *(_DWORD *)(v53 + 56);
      if ( v55 <= v56 )
      {
        if ( *(_DWORD *)(v53 + 56) )
          memmove(*(void **)(v52 + 48), *(const void **)(v53 + 48), 8 * v55);
      }
      else
      {
        if ( v55 > *(unsigned int *)(v52 + 60) )
        {
          v58 = 0;
          *(_DWORD *)(v52 + 56) = 0;
          sub_C8D5F0(v52 + 48, (const void *)(v52 + 64), v55, 8u, v50, v51);
          v55 = *(unsigned int *)(v53 + 56);
        }
        else
        {
          v58 = 8 * v56;
          if ( *(_DWORD *)(v52 + 56) )
          {
            memmove(*(void **)(v52 + 48), *(const void **)(v53 + 48), 8 * v56);
            v55 = *(unsigned int *)(v53 + 56);
          }
        }
        v59 = *(_QWORD *)(v53 + 48);
        v60 = 8 * v55;
        if ( v59 + v58 != v60 + v59 )
          memcpy((void *)(v58 + *(_QWORD *)(v52 + 48)), (const void *)(v59 + v58), v60 - v58);
      }
      *(_DWORD *)(v52 + 56) = v57;
    }
    *(_BYTE *)(v52 + 44) = *(_BYTE *)(v53 + 44) & 4 | *(_BYTE *)(v52 + 44) & 0xFB;
    *(_BYTE *)(v53 + 44) &= ~4u;
    *a1 = v52;
    goto LABEL_57;
  }
  v66 = *(const char **)(*a2 + 64);
  if ( v66 )
  {
    if ( !*v66 )
      goto LABEL_56;
    v67 = *(_DWORD *)(v52 + 56);
    do
    {
      v68 = v67;
      v69 = v67 + 1LL;
      if ( v69 > *(unsigned int *)(v52 + 60) )
      {
        v94 = v66;
        sub_C8D5F0(v52 + 48, (const void *)(v52 + 64), v67 + 1LL, 8u, (__int64)v66, v69);
        v68 = *(unsigned int *)(v52 + 56);
        v66 = v94;
      }
      *(_QWORD *)(*(_QWORD *)(v52 + 48) + 8 * v68) = v66;
      v67 = *(_DWORD *)(v52 + 56) + 1;
      *(_DWORD *)(v52 + 56) = v67;
      v66 += strlen(v66) + 1;
    }
    while ( *v66 );
    if ( *(_BYTE *)(v14 + 44) != 4 || *(_QWORD *)(*a2 + 64) )
      goto LABEL_56;
    goto LABEL_64;
  }
  if ( *(_BYTE *)(v14 + 44) == 4 )
  {
LABEL_64:
    v71 = *(unsigned int *)(v52 + 56);
    if ( v71 + 1 > (unsigned __int64)*(unsigned int *)(v52 + 60) )
    {
      sub_C8D5F0(v52 + 48, (const void *)(v52 + 64), v71 + 1, 8u, (__int64)v66, v51);
      v71 = *(unsigned int *)(v52 + 56);
    }
    *(_QWORD *)(*(_QWORD *)(v52 + 48) + 8 * v71) = byte_3F871B3;
    ++*(_DWORD *)(v52 + 56);
  }
LABEL_56:
  *a1 = v52;
LABEL_57:
  v70 = v95;
  if ( v95 )
  {
    sub_314D410(v95);
    j_j___libc_free_0(v70);
  }
  return a1;
}
