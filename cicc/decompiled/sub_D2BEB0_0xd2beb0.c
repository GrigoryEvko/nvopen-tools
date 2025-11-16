// Function: sub_D2BEB0
// Address: 0xd2beb0
//
__int64 __fastcall sub_D2BEB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 i; // rcx
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rdx
  bool v13; // r12
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  _QWORD *v16; // r15
  _QWORD *v17; // r14
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v24; // r14
  _QWORD *v25; // rdx
  _QWORD *v26; // r14
  __int64 v27; // r15
  _QWORD *v28; // r13
  unsigned __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  __int64 *v36; // r12
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  _QWORD *v42; // r14
  __int64 v43; // rdx
  __int64 *v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rsi
  __int64 v50; // r12
  int v51; // r15d
  __int64 v52; // r12
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r14
  __int64 *v57; // rax
  __int64 *v58; // rbx
  _QWORD *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rdx
  _QWORD *v62; // r14
  int v63; // r14d
  __int64 v64; // r12
  __int64 v65; // r15
  __int64 v66; // r10
  __int64 v67; // rdx
  __int64 v68; // rdi
  __int64 v69; // rax
  char *v70; // r8
  __int64 v71; // rcx
  _QWORD *v72; // r11
  unsigned __int64 v73; // r10
  char *v74; // rsi
  int v75; // r12d
  __int64 *v76; // rsi
  __int64 v77; // [rsp+0h] [rbp-90h]
  unsigned __int64 v78; // [rsp+8h] [rbp-88h]
  __int64 v79; // [rsp+10h] [rbp-80h]
  int v80; // [rsp+10h] [rbp-80h]
  unsigned __int8 v81; // [rsp+1Ch] [rbp-74h]
  __int64 *v82; // [rsp+20h] [rbp-70h]
  _QWORD *v83; // [rsp+28h] [rbp-68h]
  __int64 v84; // [rsp+28h] [rbp-68h]
  char *v85; // [rsp+28h] [rbp-68h]
  _QWORD *v86; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v87; // [rsp+38h] [rbp-58h] BYREF
  __int64 *v88; // [rsp+40h] [rbp-50h] BYREF
  __int64 v89; // [rsp+48h] [rbp-48h]
  _QWORD v90[8]; // [rsp+50h] [rbp-40h] BYREF

  v5 = a1;
  v78 = sub_D29010(a1, a2);
  v86 = (_QWORD *)sub_D23C40(a1, v78);
  v6 = (__int64 *)sub_D23C40(a1, v78);
  v79 = (__int64)v6;
  if ( v6 )
    v79 = *v6;
  v7 = a2 + 72;
  v82 = (__int64 *)sub_D29770(a1, a3);
  v8 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 == v8 )
  {
    i = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v8 + 32);
      if ( i != v8 + 24 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v7 == v8 )
        break;
      if ( !v8 )
        BUG();
    }
  }
  v10 = 0x8000000000041LL;
  while ( 1 )
  {
    if ( v8 == v7 )
    {
      v81 = 0;
      v13 = 0;
      goto LABEL_25;
    }
    if ( !i )
      BUG();
    if ( (unsigned __int8)(*(_BYTE *)(i - 24) - 34) <= 0x33u )
    {
      if ( _bittest64(&v10, (unsigned int)*(unsigned __int8 *)(i - 24) - 34) )
      {
        v11 = *(_QWORD *)(i - 56);
        if ( v11 )
        {
          if ( !*(_BYTE *)v11 && a3 == v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(i + 56) )
            break;
        }
      }
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v8 + 32) )
    {
      v12 = v8 - 24;
      if ( !v8 )
        v12 = 0;
      if ( i != v12 + 48 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v7 == v8 )
        break;
      if ( !v8 )
        BUG();
    }
  }
  v81 = 1;
  v13 = a3 == v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(i + 56);
LABEL_25:
  v87 = 0;
  v77 = (__int64)(v82 + 3);
  v14 = sub_D23BF0((__int64)(v82 + 3));
  v16 = v15;
  v17 = v14;
  v18 = sub_D23C30((__int64)(v82 + 3));
  if ( v17 == (_QWORD *)v18 )
    goto LABEL_38;
  while ( !v13 || (*v17 & 4) == 0 || v86 != (_QWORD *)sub_D23C40(a1, *v17 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    do
      ++v17;
    while ( v16 != v17 && ((*v17 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL)) );
    if ( v17 == (_QWORD *)v18 )
    {
      v5 = a1;
      goto LABEL_38;
    }
  }
  v5 = a1;
  v87 = v86;
  sub_D249B0((__int64)(v86 + 1), (__int64)v82, v19, v20, v21, v22);
  v24 = (__int64)v86;
  if ( !v86 )
  {
LABEL_38:
    v83 = sub_D23BF0(v77);
    v26 = v25;
    v27 = sub_D23C30(v77);
    if ( v83 != (_QWORD *)v27 )
    {
      v28 = v83;
      while ( 1 )
      {
        v29 = *v28 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = (_QWORD *)sub_D23C40(v5, v29);
        if ( v30 )
          v30 = (_QWORD *)*v30;
        if ( v30 == (_QWORD *)v79 )
          break;
        for ( ++v28; v28 != v26; ++v28 )
        {
          if ( (*v28 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v28 & 0xFFFFFFFFFFFFFFF8LL) )
            break;
        }
        if ( (_QWORD *)v27 == v28 )
          goto LABEL_49;
      }
      v61 = *(_QWORD *)(v5 + 208);
      *(_QWORD *)(v5 + 288) += 32LL;
      v90[0] = v82;
      v62 = (_QWORD *)((v61 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v89 = 0x100000001LL;
      v88 = v90;
      if ( *(_QWORD *)(v5 + 216) >= (unsigned __int64)(v62 + 4) && v61 )
      {
        *(_QWORD *)(v5 + 208) = v62 + 4;
      }
      else
      {
        v29 = 32;
        v62 = (_QWORD *)sub_9D1E70(v5 + 208, 32, 32, 3);
      }
      *v62 = v79;
      v62[1] = v62 + 3;
      v62[2] = 0x100000000LL;
      if ( (_DWORD)v89 )
      {
        v29 = (unsigned __int64)&v88;
        sub_D230A0((__int64)(v62 + 1), (char **)&v88, (unsigned int)v89, v31, v32, v33);
      }
      v87 = v62;
      if ( v88 != v90 )
        _libc_free(v88, v29);
      if ( v13 )
        v63 = *(_DWORD *)sub_D25AF0(v79 + 56, (__int64 *)&v86);
      else
        v63 = *(_DWORD *)(v79 + 64) >> 1;
      v64 = (__int64)v87;
      v65 = 8LL * v63;
      v66 = *(unsigned int *)(v79 + 16);
      v67 = *(_QWORD *)(v79 + 8);
      v68 = v79 + 8;
      v69 = 8 * v66;
      v70 = (char *)(v67 + v65);
      LODWORD(v71) = v66;
      v72 = (_QWORD *)(v67 + 8 * v66);
      if ( (_QWORD *)(v67 + v65) == v72 )
      {
        sub_D24960(v68, (__int64)v87, v67, v66, (__int64)v70, v63);
      }
      else
      {
        v73 = v66 + 1;
        if ( v73 > *(unsigned int *)(v79 + 20) )
        {
          sub_C8D5F0(v68, (const void *)(v79 + 24), v73, 8u, (__int64)v70, v63);
          v67 = *(_QWORD *)(v79 + 8);
          v71 = *(unsigned int *)(v79 + 16);
          v69 = 8 * v71;
          v70 = (char *)(v67 + v65);
          v72 = (_QWORD *)(v67 + 8 * v71);
        }
        v74 = (char *)(v67 + v69 - 8);
        if ( v72 )
        {
          *v72 = *(_QWORD *)v74;
          v67 = *(_QWORD *)(v79 + 8);
          v71 = *(unsigned int *)(v79 + 16);
          v69 = 8 * v71;
          v74 = (char *)(v67 + 8 * v71 - 8);
        }
        if ( v74 != v70 )
        {
          v85 = v70;
          memmove((void *)(v67 + v69 - (v74 - v70)), v70, v74 - v70);
          v70 = v85;
          LODWORD(v71) = *(_DWORD *)(v79 + 16);
        }
        *(_DWORD *)(v79 + 16) = v71 + 1;
        *(_QWORD *)v70 = v64;
      }
      v75 = *(_DWORD *)(v79 + 16);
      if ( v63 < v75 )
      {
        do
        {
          v76 = (__int64 *)(v65 + *(_QWORD *)(v79 + 8));
          v65 += 8;
          *(_DWORD *)sub_D25AF0(v79 + 56, v76) = v63++;
        }
        while ( v75 != v63 );
      }
    }
LABEL_49:
    v24 = (__int64)v87;
    if ( !v87 )
    {
      v34 = *(_QWORD *)(v5 + 336);
      *(_QWORD *)(v5 + 416) += 136LL;
      v35 = ((v34 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 136;
      if ( *(_QWORD *)(v5 + 344) >= v35 && v34 )
      {
        *(_QWORD *)(v5 + 336) = v35;
        v36 = (__int64 *)((v34 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        v36 = (__int64 *)sub_9D1E70(v5 + 336, 136, 136, 3);
      }
      v37 = v5;
      sub_D23F30(v36, v5);
      v88 = v90;
      *(_QWORD *)(v5 + 288) += 32LL;
      v90[0] = v82;
      v89 = 0x100000001LL;
      v41 = *(_QWORD *)(v5 + 208);
      v42 = (_QWORD *)((v41 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v43 = (__int64)(v42 + 4);
      if ( *(_QWORD *)(v5 + 216) >= (unsigned __int64)(v42 + 4) && v41 )
      {
        *(_QWORD *)(v5 + 208) = v43;
      }
      else
      {
        v37 = 32;
        v42 = (_QWORD *)sub_9D1E70(v5 + 208, 32, 32, 3);
      }
      *v42 = v36;
      v42[1] = v42 + 3;
      v42[2] = 0x100000000LL;
      if ( (_DWORD)v89 )
      {
        v37 = (__int64)&v88;
        sub_D230A0((__int64)(v42 + 1), (char **)&v88, v43, v38, v39, v40);
      }
      v87 = v42;
      if ( v88 != v90 )
        _libc_free(v88, v37);
      v44 = v36;
      *(_DWORD *)sub_D25AF0((__int64)(v36 + 7), (__int64 *)&v87) = 0;
      sub_D24960((__int64)(v36 + 1), (__int64)v87, v45, v46, v47, v48);
      sub_D248B0(&v88, (__int64 *)(v5 + 576), v79);
      v49 = *(_QWORD *)(v5 + 432);
      v50 = *(int *)(v90[0] + 8LL);
      v88 = v44;
      v51 = v50;
      v52 = 8 * v50;
      sub_D23810(v5 + 432, (char *)(v52 + v49), (__int64 *)&v88, v53, v54, v55);
      v80 = *(_DWORD *)(v5 + 440);
      if ( v51 < v80 )
      {
        v56 = v5 + 576;
        v84 = v5;
        do
        {
          v58 = (__int64 *)(v52 + *(_QWORD *)(v84 + 432));
          if ( (unsigned __int8)sub_D24D10(v56, v58, &v88) )
          {
            v57 = v88 + 1;
          }
          else
          {
            v59 = sub_D27750(v56, v58, v88);
            v60 = *v58;
            v57 = v59 + 1;
            *(_DWORD *)v57 = 0;
            *(v57 - 1) = v60;
          }
          *(_DWORD *)v57 = v51;
          v52 += 8;
          ++v51;
        }
        while ( v80 != v51 );
        v5 = v84;
      }
      v24 = (__int64)v87;
    }
  }
  v88 = v82;
  *sub_D25E90(v5 + 304, (__int64 *)&v88) = v24;
  return sub_D25590(v78 + 24, (__int64)v82, v81);
}
