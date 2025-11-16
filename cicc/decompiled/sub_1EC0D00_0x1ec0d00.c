// Function: sub_1EC0D00
// Address: 0x1ec0d00
//
void __fastcall sub_1EC0D00(
        _QWORD *a1,
        _QWORD *a2,
        unsigned int *a3,
        unsigned int *a4,
        unsigned int *a5,
        unsigned int *a6)
{
  __int64 *v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 (*v11)(void); // rax
  _QWORD *v12; // r15
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rsi
  int v17; // edx
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r10
  __int64 v21; // r13
  __int64 v22; // rbx
  __int64 v23; // r15
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 (*v26)(); // rcx
  __int64 v27; // rax
  __int64 (*v28)(); // rcx
  __int64 **v29; // r14
  __int64 v30; // rax
  __int64 v31; // r15
  __m128i v32; // xmm0
  __int64 v33; // rax
  _QWORD *v34; // r13
  _QWORD *v35; // r12
  _QWORD *v36; // rdi
  int v37; // eax
  int v38; // edi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // [rsp+8h] [rbp-2D8h]
  _QWORD *v51; // [rsp+48h] [rbp-298h]
  __int64 *v52; // [rsp+50h] [rbp-290h]
  int v54; // [rsp+64h] [rbp-27Ch] BYREF
  __int64 v55; // [rsp+68h] [rbp-278h] BYREF
  __m128i v56; // [rsp+70h] [rbp-270h] BYREF
  _QWORD v57[2]; // [rsp+80h] [rbp-260h] BYREF
  _QWORD *v58; // [rsp+90h] [rbp-250h]
  _QWORD v59[6]; // [rsp+A0h] [rbp-240h] BYREF
  void *v60; // [rsp+D0h] [rbp-210h] BYREF
  int v61; // [rsp+D8h] [rbp-208h]
  char v62; // [rsp+DCh] [rbp-204h]
  __int64 v63; // [rsp+E0h] [rbp-200h]
  __m128i v64; // [rsp+E8h] [rbp-1F8h]
  __int64 v65; // [rsp+F8h] [rbp-1E8h]
  char *v66; // [rsp+100h] [rbp-1E0h]
  const char *v67; // [rsp+108h] [rbp-1D8h]
  __int64 v68; // [rsp+110h] [rbp-1D0h]
  char v69; // [rsp+120h] [rbp-1C0h]
  _BYTE *v70; // [rsp+128h] [rbp-1B8h]
  __int64 v71; // [rsp+130h] [rbp-1B0h]
  _BYTE v72[356]; // [rsp+138h] [rbp-1A8h] BYREF
  int v73; // [rsp+29Ch] [rbp-44h]
  __int64 v74; // [rsp+2A0h] [rbp-40h]

  *a3 = 0;
  *a4 = 0;
  *a5 = 0;
  *a6 = 0;
  v52 = (__int64 *)a2[2];
  if ( (__int64 *)a2[1] != v52 )
  {
    v6 = (__int64 *)a2[1];
    do
    {
      v7 = *v6++;
      sub_1EC0D00(a1, v7, &v54, &v55, &v56, &v60);
      *a3 += v54;
      *a4 += v55;
      *a5 += v56.m128i_i32[0];
      *a6 += (unsigned int)v60;
    }
    while ( v52 != v6 );
  }
  v8 = 0;
  v9 = a1[85];
  v10 = *(_QWORD *)(v9 + 56);
  v11 = *(__int64 (**)(void))(**(_QWORD **)(v9 + 16) + 40LL);
  if ( v11 != sub_1D00B00 )
    v8 = v11();
  v12 = (_QWORD *)a2[4];
  v51 = (_QWORD *)a2[5];
  if ( v12 != v51 )
  {
    while ( 1 )
    {
      v13 = a1[103];
      v14 = *(_DWORD *)(v13 + 256);
      if ( !v14 )
        goto LABEL_8;
      v15 = *v12;
      v16 = *(_QWORD *)(v13 + 240);
      v17 = v14 - 1;
      v18 = v17 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( *v12 == *v19 )
      {
LABEL_11:
        if ( (_QWORD *)v19[1] != a2 )
          goto LABEL_8;
        v21 = *(_QWORD *)(v15 + 32);
        v22 = v15 + 24;
        if ( v22 == v21 )
          goto LABEL_8;
        v45 = v12;
        v23 = v8;
        v24 = v21;
        do
        {
          while ( 1 )
          {
            v25 = *(_QWORD *)v23;
            v26 = *(__int64 (**)())(*(_QWORD *)v23 + 48LL);
            if ( v26 != sub_1E1C810 )
            {
              if ( ((unsigned int (__fastcall *)(__int64, __int64, int *))v26)(v23, v24, &v54)
                && *(_BYTE *)(*(_QWORD *)(v10 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v10 + 32) + v54) + 21) )
              {
                ++*a3;
                goto LABEL_16;
              }
              v25 = *(_QWORD *)v23;
            }
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, void **, int *))(v25 + 72))(v23, v24, &v60, &v54)
              && *(_BYTE *)(*(_QWORD *)(v10 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v10 + 32) + v54) + 21) )
            {
              ++*a4;
              goto LABEL_16;
            }
            v27 = *(_QWORD *)v23;
            v28 = *(__int64 (**)())(*(_QWORD *)v23 + 80LL);
            if ( v28 != sub_1EBAF80 )
            {
              if ( ((unsigned int (__fastcall *)(__int64, __int64, int *))v28)(v23, v24, &v54)
                && *(_BYTE *)(*(_QWORD *)(v10 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v10 + 32) + v54) + 21) )
              {
                ++*a5;
                goto LABEL_16;
              }
              v27 = *(_QWORD *)v23;
            }
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, void **, int *))(v27 + 104))(v23, v24, &v60, &v54)
              && *(_BYTE *)(*(_QWORD *)(v10 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v10 + 32) + v54) + 21) )
            {
              ++*a6;
            }
LABEL_16:
            if ( !v24 )
              BUG();
            if ( (*(_BYTE *)v24 & 4) == 0 )
              break;
            v24 = *(_QWORD *)(v24 + 8);
            if ( v22 == v24 )
              goto LABEL_28;
          }
          while ( (*(_BYTE *)(v24 + 46) & 8) != 0 )
            v24 = *(_QWORD *)(v24 + 8);
          v24 = *(_QWORD *)(v24 + 8);
        }
        while ( v22 != v24 );
LABEL_28:
        v8 = v23;
        v12 = v45 + 1;
        if ( v51 == v45 + 1 )
          break;
      }
      else
      {
        v37 = 1;
        while ( v20 != -8 )
        {
          v38 = v37 + 1;
          v18 = v17 & (v37 + v18);
          v19 = (__int64 *)(v16 + 16LL * v18);
          v20 = *v19;
          if ( v15 == *v19 )
            goto LABEL_11;
          v37 = v38;
        }
LABEL_8:
        if ( v51 == ++v12 )
          break;
      }
    }
  }
  if ( *a3 || *a4 || *a5 || *a6 )
  {
    v29 = (__int64 **)a1[104];
    v30 = sub_15E0530(**v29);
    if ( sub_1602790(v30)
      || (v39 = sub_15E0530(**v29),
          v40 = sub_16033E0(v39),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v40 + 48LL))(v40)) )
    {
      v31 = *(_QWORD *)a2[4];
      sub_1E299E0(&v55, (__int64)a2);
      sub_15C9090((__int64)&v56, &v55);
      v32 = _mm_loadu_si128(&v56);
      v33 = **(_QWORD **)(v31 + 56);
      v62 = 2;
      v61 = 15;
      v63 = v33;
      v68 = 15;
      v65 = v57[0];
      v66 = "regalloc";
      v67 = "LoopSpillReload";
      v71 = 0x400000000LL;
      v69 = 0;
      v70 = v72;
      v72[352] = 0;
      v73 = -1;
      v74 = v31;
      v60 = &unk_49FC050;
      v64 = v32;
      if ( v55 )
        sub_161E7C0((__int64)&v55, v55);
      if ( *a5 )
      {
        sub_15C9C50((__int64)&v56, "NumSpills", 9, *a5);
        v44 = sub_1E3AF10((__int64)&v60, (__int64)&v56);
        sub_15CAB20(v44, " spills ", 8u);
        if ( v58 != v59 )
          j_j___libc_free_0(v58, v59[0] + 1LL);
        if ( (_QWORD *)v56.m128i_i64[0] != v57 )
          j_j___libc_free_0(v56.m128i_i64[0], v57[0] + 1LL);
      }
      if ( *a6 )
      {
        sub_15C9C50((__int64)&v56, "NumFoldedSpills", 15, *a6);
        v43 = sub_1E3AF10((__int64)&v60, (__int64)&v56);
        sub_15CAB20(v43, " folded spills ", 0xFu);
        if ( v58 != v59 )
          j_j___libc_free_0(v58, v59[0] + 1LL);
        if ( (_QWORD *)v56.m128i_i64[0] != v57 )
          j_j___libc_free_0(v56.m128i_i64[0], v57[0] + 1LL);
      }
      if ( *a3 )
      {
        sub_15C9C50((__int64)&v56, "NumReloads", 10, *a3);
        v42 = sub_1E3AF10((__int64)&v60, (__int64)&v56);
        sub_15CAB20(v42, " reloads ", 9u);
        if ( v58 != v59 )
          j_j___libc_free_0(v58, v59[0] + 1LL);
        if ( (_QWORD *)v56.m128i_i64[0] != v57 )
          j_j___libc_free_0(v56.m128i_i64[0], v57[0] + 1LL);
      }
      if ( *a4 )
      {
        sub_15C9C50((__int64)&v56, "NumFoldedReloads", 16, *a4);
        v41 = sub_1E3AF10((__int64)&v60, (__int64)&v56);
        sub_15CAB20(v41, " folded reloads ", 0x10u);
        if ( v58 != v59 )
          j_j___libc_free_0(v58, v59[0] + 1LL);
        if ( (_QWORD *)v56.m128i_i64[0] != v57 )
          j_j___libc_free_0(v56.m128i_i64[0], v57[0] + 1LL);
      }
      sub_15CAB20((__int64)&v60, "generated in loop", 0x11u);
      sub_1E36D90(v29, (__int64)&v60);
      v34 = v70;
      v60 = &unk_49ECF68;
      v35 = &v70[88 * (unsigned int)v71];
      if ( v70 != (_BYTE *)v35 )
      {
        do
        {
          v35 -= 11;
          v36 = (_QWORD *)v35[4];
          if ( v36 != v35 + 6 )
            j_j___libc_free_0(v36, v35[6] + 1LL);
          if ( (_QWORD *)*v35 != v35 + 2 )
            j_j___libc_free_0(*v35, v35[2] + 1LL);
        }
        while ( v34 != v35 );
        v35 = v70;
      }
      if ( v35 != (_QWORD *)v72 )
        _libc_free((unsigned __int64)v35);
    }
  }
}
