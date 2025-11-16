// Function: sub_2D5D3F0
// Address: 0x2d5d3f0
//
__int64 __fastcall sub_2D5D3F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 *v11; // r15
  __int64 v12; // r13
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  _QWORD *v16; // rdx
  __int64 *v17; // r12
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int8 v20; // al
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdi
  unsigned __int8 v24; // al
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r9
  unsigned __int8 v28; // r8
  __int64 *v29; // rax
  __int64 v30; // rax
  int v31; // eax
  unsigned __int64 v32; // r15
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned __int64 v35; // rbx
  __int64 v36; // r12
  unsigned __int64 v37; // r13
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  int v40; // eax
  unsigned __int64 v41; // r15
  __int64 v42; // rcx
  __int64 v43; // rax
  unsigned __int64 v44; // rbx
  __int64 v45; // r12
  __int64 v46; // rbx
  unsigned __int64 v47; // r13
  unsigned __int64 v48; // rdi
  __int64 *v49; // [rsp+0h] [rbp-150h]
  __int64 *v50; // [rsp+0h] [rbp-150h]
  __int64 v51; // [rsp+8h] [rbp-148h]
  __int64 v52; // [rsp+8h] [rbp-148h]
  unsigned __int8 v54; // [rsp+1Fh] [rbp-131h]
  __int64 v55; // [rsp+28h] [rbp-128h]
  __int64 *v56; // [rsp+30h] [rbp-120h]
  unsigned __int8 v57; // [rsp+30h] [rbp-120h]
  unsigned __int8 v58; // [rsp+30h] [rbp-120h]
  __int64 *v59; // [rsp+38h] [rbp-118h]
  __int64 *v60; // [rsp+40h] [rbp-110h]
  unsigned __int8 v61; // [rsp+40h] [rbp-110h]
  __int64 v62; // [rsp+50h] [rbp-100h]
  __int64 v63; // [rsp+58h] [rbp-F8h]
  char v64[48]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 *v65; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+98h] [rbp-B8h]
  _BYTE v67[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a1 + 792);
  v55 = v5 + 152LL * *(unsigned int *)(a1 + 800);
  if ( v5 == v55 )
    return 0;
  v63 = *(_QWORD *)(a1 + 792);
  v6 = a1;
  v54 = 0;
  do
  {
    v65 = (__int64 *)v67;
    v66 = 0x1000000000LL;
    v7 = *(_QWORD *)(v63 + 8);
    v8 = (__int64 *)(v7 + 8LL * *(unsigned int *)(v63 + 16));
    if ( v8 == (__int64 *)v7 )
      goto LABEL_13;
    v9 = v6 + 376;
    v10 = v6;
    v11 = *(__int64 **)(v63 + 8);
    v62 = v9;
    while ( 1 )
    {
LABEL_5:
      while ( 1 )
      {
        v12 = *v11;
        if ( !*(_BYTE *)(v10 + 404) )
          break;
        v13 = *(_QWORD **)(v10 + 384);
        v14 = &v13[*(unsigned int *)(v10 + 396)];
        if ( v13 == v14 )
          goto LABEL_16;
        while ( v12 != *v13 )
        {
          if ( v14 == ++v13 )
            goto LABEL_16;
        }
LABEL_10:
        if ( v8 == ++v11 )
          goto LABEL_11;
      }
      if ( sub_C8CA60(v62, *v11) )
        goto LABEL_10;
LABEL_16:
      if ( *(_BYTE *)v12 != 69 )
        goto LABEL_10;
      v16 = (*(_BYTE *)(v12 + 7) & 0x40) != 0
          ? *(_QWORD **)(v12 - 8)
          : (_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
      if ( *(_QWORD *)v63 != *v16 )
        goto LABEL_10;
      v17 = v65;
      v18 = (unsigned int)v66;
      v56 = &v65[(unsigned int)v66];
      if ( v56 != v65 )
        break;
LABEL_28:
      if ( v18 + 1 > (unsigned __int64)HIDWORD(v66) )
      {
        sub_C8D5F0((__int64)&v65, v67, v18 + 1, 8u, a5, v7);
        v18 = (unsigned int)v66;
      }
      ++v11;
      v65[v18] = v12;
      LODWORD(v66) = v66 + 1;
      if ( v8 == v11 )
        goto LABEL_11;
    }
    v60 = v8;
    v59 = v11;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v10 + 824);
      if ( !v19 )
      {
        v30 = sub_22077B0(0x80u);
        v19 = v30;
        if ( v30 )
        {
          *(_QWORD *)(v30 + 96) = 0;
          *(_QWORD *)v30 = v30 + 16;
          *(_QWORD *)(v30 + 8) = 0x100000000LL;
          *(_QWORD *)(v30 + 24) = v30 + 40;
          *(_QWORD *)(v30 + 32) = 0x600000000LL;
          *(_BYTE *)(v30 + 112) = 0;
          *(_QWORD *)(v30 + 104) = a2;
          v31 = *(_DWORD *)(a2 + 92);
          *(_DWORD *)(v19 + 116) = 0;
          *(_DWORD *)(v19 + 120) = v31;
          sub_B1F440(v19);
        }
        v32 = *(_QWORD *)(v10 + 824);
        *(_QWORD *)(v10 + 824) = v19;
        if ( v32 )
          break;
      }
      v20 = sub_B19DB0(v19, v12, *v17);
      if ( v20 )
        goto LABEL_57;
LABEL_24:
      v23 = *(_QWORD *)(v10 + 824);
      if ( !v23 )
      {
        v39 = sub_22077B0(0x80u);
        v23 = v39;
        if ( v39 )
        {
          *(_QWORD *)(v39 + 96) = 0;
          *(_QWORD *)v39 = v39 + 16;
          *(_QWORD *)(v39 + 8) = 0x100000000LL;
          *(_QWORD *)(v39 + 24) = v39 + 40;
          *(_QWORD *)(v39 + 32) = 0x600000000LL;
          *(_BYTE *)(v39 + 112) = 0;
          *(_QWORD *)(v39 + 104) = a2;
          v40 = *(_DWORD *)(a2 + 92);
          *(_DWORD *)(v23 + 116) = 0;
          *(_DWORD *)(v23 + 120) = v40;
          sub_B1F440(v23);
        }
        v41 = *(_QWORD *)(v10 + 824);
        *(_QWORD *)(v10 + 824) = v23;
        if ( v41 )
        {
          v42 = *(_QWORD *)(v41 + 24);
          v43 = *(unsigned int *)(v41 + 32);
          v44 = v42 + 8 * v43;
          if ( v42 != v44 )
          {
            v52 = v12;
            v50 = v17;
            v45 = v42 + 8 * v43;
            v46 = *(_QWORD *)(v41 + 24);
            do
            {
              v47 = *(_QWORD *)(v45 - 8);
              v45 -= 8;
              if ( v47 )
              {
                v48 = *(_QWORD *)(v47 + 24);
                if ( v48 != v47 + 40 )
                  _libc_free(v48);
                j_j___libc_free_0(v47);
              }
            }
            while ( v46 != v45 );
            v12 = v52;
            v17 = v50;
            v44 = *(_QWORD *)(v41 + 24);
          }
          if ( v44 != v41 + 40 )
            _libc_free(v44);
          if ( *(_QWORD *)v41 != v41 + 16 )
            _libc_free(*(_QWORD *)v41);
          j_j___libc_free_0(v41);
          v23 = *(_QWORD *)(v10 + 824);
        }
      }
      v24 = sub_B19DB0(v23, *v17, v12);
      if ( v24 )
      {
        v57 = v24;
        v8 = v60;
        sub_2D594F0(v12, *v17, (__int64 *)(v10 + 840), *(unsigned __int8 *)(v10 + 832), a5, v7);
        v28 = v57;
        if ( *(_BYTE *)(v10 + 404) )
        {
          v29 = *(__int64 **)(v10 + 384);
          v26 = *(unsigned int *)(v10 + 396);
          v25 = &v29[v26];
          if ( v29 != v25 )
          {
            while ( v12 != *v29 )
            {
              if ( v25 == ++v29 )
                goto LABEL_59;
            }
LABEL_38:
            v61 = v28;
            v11 = v59 + 1;
            sub_B43D10((_QWORD *)v12);
            a5 = v61;
            v54 = v61;
            if ( v8 != v59 + 1 )
              goto LABEL_5;
            goto LABEL_11;
          }
LABEL_59:
          if ( (unsigned int)v26 < *(_DWORD *)(v10 + 392) )
          {
            *(_DWORD *)(v10 + 396) = v26 + 1;
            *v25 = v12;
            ++*(_QWORD *)(v10 + 376);
            goto LABEL_38;
          }
        }
        sub_C8CC70(v62, v12, (__int64)v25, v26, v57, v27);
        v28 = v57;
        goto LABEL_38;
      }
      if ( v56 == ++v17 )
      {
        v8 = v60;
        v11 = v59;
        v18 = (unsigned int)v66;
        goto LABEL_28;
      }
    }
    v33 = *(_QWORD *)(v32 + 24);
    v34 = *(unsigned int *)(v32 + 32);
    v35 = v33 + 8 * v34;
    if ( v33 != v35 )
    {
      v51 = v12;
      v49 = v17;
      v36 = v33 + 8 * v34;
      do
      {
        v37 = *(_QWORD *)(v36 - 8);
        v36 -= 8;
        if ( v37 )
        {
          v38 = *(_QWORD *)(v37 + 24);
          if ( v38 != v37 + 40 )
            _libc_free(v38);
          j_j___libc_free_0(v37);
        }
      }
      while ( v33 != v36 );
      v12 = v51;
      v17 = v49;
      v35 = *(_QWORD *)(v32 + 24);
    }
    if ( v35 != v32 + 40 )
      _libc_free(v35);
    if ( *(_QWORD *)v32 != v32 + 16 )
      _libc_free(*(_QWORD *)v32);
    j_j___libc_free_0(v32);
    v20 = sub_B19DB0(*(_QWORD *)(v10 + 824), v12, *v17);
    if ( !v20 )
      goto LABEL_24;
LABEL_57:
    v58 = v20;
    v8 = v60;
    sub_2D594F0(*v17, v12, (__int64 *)(v10 + 840), *(unsigned __int8 *)(v10 + 832), v21, v22);
    v11 = v59 + 1;
    sub_BED950((__int64)v64, v62, *v17);
    sub_B43D10((_QWORD *)*v17);
    *v17 = v12;
    v54 = v58;
    if ( v60 != v59 + 1 )
      goto LABEL_5;
LABEL_11:
    v6 = v10;
    if ( v65 != (__int64 *)v67 )
      _libc_free((unsigned __int64)v65);
LABEL_13:
    v63 += 152;
  }
  while ( v55 != v63 );
  return v54;
}
