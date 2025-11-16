// Function: sub_34A6040
// Address: 0x34a6040
//
void __fastcall sub_34A6040(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  bool v7; // zf
  _DWORD *v8; // r13
  char v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rdi
  signed __int64 v12; // r12
  _DWORD *v13; // rdi
  char *v14; // r13
  unsigned int *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rbx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // r9
  int v27; // esi
  unsigned int v28; // r8d
  __int64 v29; // rax
  int v30; // r10d
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // r15
  __int64 v34; // r13
  char v35; // al
  __int64 v36; // rdx
  __int64 v37; // rdx
  _QWORD *v38; // r8
  _BYTE *v39; // rdi
  __int64 v40; // rax
  _BYTE *v41; // rax
  unsigned __int64 v42; // rcx
  unsigned int *v43; // r13
  unsigned int v44; // r12d
  unsigned int *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // rcx
  __int64 v50; // rax
  unsigned int v51; // esi
  __int64 v52; // rax
  unsigned int v53; // esi
  unsigned int *v54; // rbx
  _QWORD *v55; // r12
  char v56; // r14
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  int v60; // eax
  int v61; // edx
  unsigned int *v62; // [rsp+8h] [rbp-208h]
  __int64 v63; // [rsp+18h] [rbp-1F8h]
  __int64 v64; // [rsp+20h] [rbp-1F0h]
  unsigned int *v65; // [rsp+28h] [rbp-1E8h]
  _QWORD *v66; // [rsp+30h] [rbp-1E0h]
  unsigned __int64 v67; // [rsp+38h] [rbp-1D8h]
  unsigned int *v69; // [rsp+40h] [rbp-1D0h]
  _BYTE *v71; // [rsp+50h] [rbp-1C0h] BYREF
  __int64 v72; // [rsp+58h] [rbp-1B8h]
  _BYTE v73[16]; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v74; // [rsp+70h] [rbp-1A0h] BYREF
  unsigned __int64 v75; // [rsp+78h] [rbp-198h] BYREF
  unsigned int v76; // [rsp+80h] [rbp-190h]
  char v77; // [rsp+88h] [rbp-188h] BYREF
  unsigned int v78; // [rsp+C8h] [rbp-148h]
  unsigned __int64 v79; // [rsp+D0h] [rbp-140h]
  unsigned __int64 v80; // [rsp+D8h] [rbp-138h]
  __int64 v81; // [rsp+E0h] [rbp-130h]
  char *v82; // [rsp+E8h] [rbp-128h]
  __int64 v83; // [rsp+F0h] [rbp-120h]
  char v84; // [rsp+F8h] [rbp-118h] BYREF
  int v85; // [rsp+138h] [rbp-D8h]
  __int64 v86; // [rsp+140h] [rbp-D0h]
  __int64 v87; // [rsp+148h] [rbp-C8h]
  void *base; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+158h] [rbp-B8h]
  _BYTE v90[176]; // [rsp+160h] [rbp-B0h] BYREF

  v6 = a4;
  v7 = *(_QWORD *)(a2 + 184) == 0;
  base = v90;
  v89 = 0x2000000000LL;
  if ( v7 )
  {
    v8 = *(_DWORD **)a2;
    v9 = 1;
    v10 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  }
  else
  {
    v8 = *(_DWORD **)(a2 + 168);
    v10 = a2 + 152;
    v9 = 0;
  }
  v11 = (__int64)v8;
  v12 = 0;
  while ( !v9 )
  {
    if ( v11 == v10 )
      goto LABEL_11;
    v11 = sub_220EF30(v11);
LABEL_6:
    ++v12;
  }
  if ( v11 != v10 )
  {
    v11 += 4;
    goto LABEL_6;
  }
LABEL_11:
  if ( v12 > 32 )
  {
    sub_C8D5F0((__int64)&base, v90, v12, 4u, a5, a6);
    a4 = (unsigned int)v89;
    v13 = v8;
    v14 = (char *)base + 4 * (unsigned int)v89;
  }
  else
  {
    v13 = v8;
    v14 = v90;
  }
  while ( 2 )
  {
    if ( !v9 )
    {
      if ( v13 == (_DWORD *)v10 )
        break;
      if ( v14 )
        *(_DWORD *)v14 = v13[8];
      v13 = (_DWORD *)sub_220EF30((__int64)v13);
      goto LABEL_17;
    }
    if ( v13 != (_DWORD *)v10 )
    {
      if ( v14 )
        *(_DWORD *)v14 = *v13;
      ++v13;
LABEL_17:
      v14 += 4;
      continue;
    }
    break;
  }
  v15 = (unsigned int *)base;
  LODWORD(v89) = v12 + v89;
  if ( (unsigned int)v89 > 1uLL )
  {
    qsort(base, (4LL * (unsigned int)v89) >> 2, 4u, (__compar_fn_t)sub_2F60C30);
    v15 = (unsigned int *)base;
  }
  sub_34A3D10((__int64)&v74, a3, (_QWORD *)((unsigned __int64)*v15 << 32), a4, (__int64)&v74, a6);
  v81 = 0;
  v82 = &v84;
  v83 = 0x400000000LL;
  v86 = 0;
  v65 = (unsigned int *)base;
  v16 = v78;
  v87 = 0;
  v85 = -1;
  v62 = (unsigned int *)((char *)base + 4 * (unsigned int)v89);
  if ( base != v62 )
  {
LABEL_26:
    v17 = *v65;
    v18 = (unsigned __int64)(unsigned int)(v17 + 1) << 32;
    v19 = v17 << 32;
    v67 = v18;
    if ( v16 == -1 )
    {
      v24 = v79;
    }
    else if ( v19 <= v80 )
    {
LABEL_32:
      v24 = v79;
      v16 = v78;
      if ( v19 >= v79 )
      {
        v16 = -(int)v79;
        v78 = -(int)v79;
      }
    }
    else
    {
      while ( 1 )
      {
        v20 = v75 + 16LL * v76 - 16;
        v21 = *(_DWORD *)(v20 + 12) + 1;
        *(_DWORD *)(v20 + 12) = v21;
        v22 = v76;
        if ( v21 == *(_DWORD *)(v75 + 16LL * v76 - 8) )
        {
          v51 = *(_DWORD *)(v74 + 192);
          if ( v51 )
          {
            sub_F03D40((__int64 *)&v75, v51);
            v22 = v76;
          }
        }
        if ( !(_DWORD)v22 || *(_DWORD *)(v75 + 12) >= *(_DWORD *)(v75 + 8) )
          break;
        v78 = 0;
        v23 = v75 + 16 * v22 - 16;
        v79 = *(_QWORD *)(*(_QWORD *)v23 + 16LL * *(unsigned int *)(v23 + 12));
        v80 = *(_QWORD *)(*(_QWORD *)v23 + 16LL * *(unsigned int *)(v23 + 12) + 8);
        if ( v19 <= v80 )
          goto LABEL_32;
      }
      v78 = -1;
      v24 = 0;
      v16 = -1;
      v79 = 0;
      v80 = 0;
    }
    if ( v16 == -1 )
      goto LABEL_59;
    while ( 1 )
    {
LABEL_35:
      v25 = v24 + v16;
      if ( v67 <= v25 )
      {
LABEL_61:
        if ( v16 == -1 && !v24 && !v80 )
          break;
        if ( v62 == ++v65 )
          break;
        goto LABEL_26;
      }
      while ( 1 )
      {
        if ( (*(_BYTE *)(v6 + 56) & 1) != 0 )
        {
          v26 = v6 + 64;
          v27 = 3;
        }
        else
        {
          v46 = *(unsigned int *)(v6 + 72);
          v26 = *(_QWORD *)(v6 + 64);
          if ( !(_DWORD)v46 )
            goto LABEL_84;
          v27 = v46 - 1;
        }
        v28 = v27 & (37 * HIDWORD(v25));
        v29 = v26 + 32LL * v28;
        v30 = *(_DWORD *)v29;
        if ( *(_DWORD *)v29 == HIDWORD(v25) )
          goto LABEL_39;
        v60 = 1;
        while ( v30 != -1 )
        {
          v61 = v60 + 1;
          v28 = v27 & (v60 + v28);
          v29 = v26 + 32LL * v28;
          v30 = *(_DWORD *)v29;
          if ( HIDWORD(v25) == *(_DWORD *)v29 )
            goto LABEL_39;
          v60 = v61;
        }
        if ( (*(_BYTE *)(v6 + 56) & 1) != 0 )
        {
          v52 = 128;
          goto LABEL_85;
        }
        v46 = *(unsigned int *)(v6 + 72);
LABEL_84:
        v52 = 32 * v46;
LABEL_85:
        v29 = v26 + v52;
LABEL_39:
        v25 = (unsigned int)v25;
        v31 = *(_QWORD *)(v6 + 16);
        v32 = v6 + 8;
        v33 = *(_QWORD *)(v29 + 8) + 384LL * (unsigned int)v25;
        if ( v31 )
        {
          v34 = v6 + 8;
          do
          {
            while ( 1 )
            {
              v35 = sub_34A0190(v31 + 32, v33);
              v36 = *(_QWORD *)(v31 + 16);
              v25 = *(_QWORD *)(v31 + 24);
              if ( v35 )
                break;
              v34 = v31;
              v31 = *(_QWORD *)(v31 + 16);
              if ( !v36 )
                goto LABEL_44;
            }
            v31 = *(_QWORD *)(v31 + 24);
          }
          while ( v25 );
LABEL_44:
          if ( v34 != v32 && !(unsigned __int8)sub_34A0190(v33, v34 + 32) )
            v32 = v34;
        }
        v37 = *(unsigned int *)(v32 + 424);
        v72 = 0x200000000LL;
        v38 = &v71;
        v39 = v73;
        v40 = -8;
        v71 = v73;
        if ( (_DWORD)v37 )
        {
          sub_349DD80((__int64)&v71, v32 + 416, v37, v25, (__int64)&v71, v26);
          v39 = v71;
          v40 = 8LL * (unsigned int)v72 - 8;
        }
        v41 = &v39[v40];
        v69 = (unsigned int *)(v41 + 4);
        if ( *(_QWORD *)(a1 + 184) )
        {
          sub_B99820(a1 + 144, v69);
          v39 = v71;
          goto LABEL_55;
        }
        v42 = *(unsigned int *)(a1 + 8);
        v43 = (unsigned int *)(*(_QWORD *)a1 + 4 * v42);
        if ( *(unsigned int **)a1 == v43 )
        {
          if ( v42 > 0x1F )
          {
            v64 = a1 + 144;
LABEL_97:
            *(_DWORD *)(a1 + 8) = 0;
            sub_B99820(v64, v69);
            v39 = v71;
            goto LABEL_55;
          }
          v44 = *((_DWORD *)v41 + 1);
        }
        else
        {
          v44 = *((_DWORD *)v41 + 1);
          v45 = *(unsigned int **)a1;
          while ( *v45 != v44 )
          {
            if ( v43 == ++v45 )
              goto LABEL_79;
          }
          if ( v43 != v45 )
            goto LABEL_55;
LABEL_79:
          if ( v42 > 0x1F )
          {
            v54 = *(unsigned int **)a1;
            v63 = v6;
            v55 = (_QWORD *)(a1 + 152);
            v64 = a1 + 144;
            do
            {
              v58 = sub_B9AB10((_QWORD *)(a1 + 144), (__int64)v55, v54);
              if ( v59 )
              {
                v56 = v58 || v55 == (_QWORD *)v59 || *v54 < *(_DWORD *)(v59 + 32);
                v66 = (_QWORD *)v59;
                v57 = sub_22077B0(0x28u);
                *(_DWORD *)(v57 + 32) = *v54;
                sub_220F040(v56, v57, v66, v55);
                ++*(_QWORD *)(a1 + 184);
              }
              ++v54;
            }
            while ( v43 != v54 );
            v6 = v63;
            goto LABEL_97;
          }
        }
        if ( v42 + 1 > *(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v42 + 1, 4u, (__int64)v38, v26);
          v43 = (unsigned int *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
        }
        *v43 = v44;
        v39 = v71;
        ++*(_DWORD *)(a1 + 8);
LABEL_55:
        if ( v39 != v73 )
          _libc_free((unsigned __int64)v39);
        v24 = v79;
        if ( v79 + v78 < v80 )
        {
          v16 = v78 + 1;
          v78 = v16;
          if ( v16 != -1 )
            goto LABEL_35;
          goto LABEL_59;
        }
        v47 = v75 + 16LL * v76 - 16;
        v48 = *(_DWORD *)(v47 + 12) + 1;
        *(_DWORD *)(v47 + 12) = v48;
        v49 = v76;
        if ( v48 == *(_DWORD *)(v75 + 16LL * v76 - 8) )
        {
          v53 = *(_DWORD *)(v74 + 192);
          if ( v53 )
          {
            sub_F03D40((__int64 *)&v75, v53);
            v49 = v76;
          }
        }
        if ( (_DWORD)v49 && *(_DWORD *)(v75 + 12) < *(_DWORD *)(v75 + 8) )
          break;
        v78 = -1;
        v24 = 0;
        v79 = 0;
        v80 = 0;
LABEL_59:
        if ( !(v80 | v24) )
          goto LABEL_63;
        v16 = -1;
        v25 = v24 + 0xFFFFFFFF;
        if ( v67 <= v24 + 0xFFFFFFFF )
          goto LABEL_61;
      }
      v78 = 0;
      v50 = v75 + 16 * v49 - 16;
      v79 = *(_QWORD *)(*(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 12));
      v24 = v79;
      v80 = *(_QWORD *)(*(_QWORD *)v50 + 16LL * *(unsigned int *)(v50 + 12) + 8);
      v16 = 0;
    }
  }
LABEL_63:
  if ( (char *)v75 != &v77 )
    _libc_free(v75);
  if ( base != v90 )
    _libc_free((unsigned __int64)base);
}
