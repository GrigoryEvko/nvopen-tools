// Function: sub_37A4450
// Address: 0x37a4450
//
unsigned __int8 *__fastcall sub_37A4450(__int64 *a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 *v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v14; // rdx
  int v15; // r9d
  __int128 v16; // rax
  __int64 v17; // r9
  __int64 v18; // r8
  unsigned int v19; // r13d
  unsigned int v20; // r14d
  unsigned int v21; // eax
  unsigned int i; // ecx
  int v23; // edx
  unsigned int v24; // ebx
  __int64 *v25; // r15
  __int64 v26; // r8
  __int64 v27; // r9
  __int16 v28; // eax^2
  __int64 v29; // r9
  __int16 v30; // ebx^2
  unsigned int v31; // ebx
  _QWORD *v32; // r15
  __int128 v33; // rax
  __int64 v34; // r9
  unsigned __int8 *v35; // r8
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned __int8 **v39; // rax
  unsigned int v40; // r13d
  unsigned int v41; // r15d
  __int64 v42; // r12
  _QWORD *v43; // rdi
  _QWORD *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r8
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 *v49; // rax
  unsigned int v50; // eax
  unsigned __int8 *v51; // rax
  void *v52; // rdi
  unsigned __int8 *v53; // r14
  __int64 v55; // rdx
  _QWORD *v56; // rdi
  __int64 v57; // rcx
  _DWORD *v58; // rax
  _DWORD *v59; // rdx
  const void *v60; // r14
  __int64 v61; // r15
  _QWORD *v62; // r13
  _QWORD *v63; // rax
  __int64 v64; // rdx
  unsigned __int8 *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int128 v69; // [rsp-10h] [rbp-150h]
  __int64 v70; // [rsp-8h] [rbp-148h]
  unsigned __int8 *v71; // [rsp+0h] [rbp-140h]
  __int64 v72; // [rsp+8h] [rbp-138h]
  unsigned int v73; // [rsp+20h] [rbp-120h]
  unsigned int v74; // [rsp+24h] [rbp-11Ch]
  int v75; // [rsp+28h] [rbp-118h]
  unsigned int v76; // [rsp+28h] [rbp-118h]
  __int128 v77; // [rsp+30h] [rbp-110h]
  __int64 v78; // [rsp+40h] [rbp-100h]
  _QWORD *v79; // [rsp+40h] [rbp-100h]
  __int64 v80; // [rsp+40h] [rbp-100h]
  __int64 v81; // [rsp+48h] [rbp-F8h]
  __int64 v82; // [rsp+48h] [rbp-F8h]
  __int64 v83; // [rsp+50h] [rbp-F0h]
  unsigned int v84; // [rsp+50h] [rbp-F0h]
  unsigned __int16 v86; // [rsp+60h] [rbp-E0h]
  __int16 v87; // [rsp+60h] [rbp-E0h]
  _QWORD *v88; // [rsp+60h] [rbp-E0h]
  __int64 v89; // [rsp+68h] [rbp-D8h]
  int v90; // [rsp+70h] [rbp-D0h]
  __int64 v91; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v92; // [rsp+88h] [rbp-B8h]
  __int64 v93; // [rsp+90h] [rbp-B0h] BYREF
  int v94; // [rsp+98h] [rbp-A8h]
  __int64 v95; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v96; // [rsp+A8h] [rbp-98h]
  __int64 v97; // [rsp+B0h] [rbp-90h] BYREF
  int v98; // [rsp+B8h] [rbp-88h]
  void *s; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v100; // [rsp+C8h] [rbp-78h]
  _QWORD v101[14]; // [rsp+D0h] [rbp-70h] BYREF

  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  LOWORD(v91) = v8;
  v92 = v9;
  if ( (_WORD)v8 )
  {
    v83 = 0;
    v86 = word_4456580[v8 - 1];
  }
  else
  {
    v86 = sub_3009970((__int64)&v91, a2, v9, a5, a6);
    v83 = v67;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v93 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v93, v10, 1);
  v11 = *a1;
  v12 = a1[1];
  v94 = *(_DWORD *)(a2 + 72);
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v11 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&s, v11, *(_QWORD *)(v12 + 64), v91, v92);
    LOWORD(v95) = v100;
    v96 = v101[0];
  }
  else
  {
    LODWORD(v95) = v13(v11, *(_QWORD *)(v12 + 64), v91, v92);
    v96 = v68;
  }
  sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v70 = v14;
  *(_QWORD *)&v16 = sub_33FAF80(a1[1], 164, (__int64)&v93, (unsigned int)v95, v96, v15, a3);
  v18 = v70;
  v77 = v16;
  if ( (_WORD)v95 )
    v74 = word_4456340[(unsigned __int16)v95 - 1];
  else
    v74 = sub_3007240((__int64)&v95);
  if ( (_WORD)v91 )
  {
    v73 = word_4456340[(unsigned __int16)v91 - 1];
    v19 = v74 - v73;
    if ( (unsigned __int16)(v91 - 176) <= 0x34u )
      goto LABEL_11;
LABEL_39:
    s = v101;
    v100 = 0x1000000000LL;
    if ( v74 > 0x10 )
    {
      sub_C8D5F0((__int64)&s, v101, v74, 4u, v18, v17);
      memset(s, 255, 4LL * v74);
      LODWORD(v100) = v74;
      v58 = s;
    }
    else
    {
      if ( v74 )
      {
        v55 = 4LL * v74;
        if ( v55 )
        {
          v56 = v101;
          if ( (unsigned int)v55 >= 8 )
          {
            v57 = (unsigned int)v55 >> 3;
            memset(v101, 0xFFu, 8 * v57);
            v56 = &v101[v57];
          }
          if ( ((4 * (_BYTE)v74) & 4) != 0 )
            *(_DWORD *)v56 = -1;
        }
      }
      LODWORD(v100) = v74;
      v58 = v101;
    }
    v59 = &v58[v73];
    if ( v59 != v58 )
    {
      do
        *v58++ = v19++;
      while ( v59 != v58 );
      v58 = s;
    }
    v60 = v58;
    v61 = (unsigned int)v100;
    v97 = 0;
    v98 = 0;
    v62 = (_QWORD *)a1[1];
    v63 = sub_33F17F0(v62, 51, (__int64)&v97, v95, v96);
    if ( v97 )
    {
      v88 = v63;
      v89 = v64;
      sub_B91220((__int64)&v97, v97);
      v63 = v88;
      v64 = v89;
    }
    v65 = (unsigned __int8 *)sub_33FCE10(
                               (__int64)v62,
                               (unsigned int)v95,
                               v96,
                               (__int64)&v93,
                               v77,
                               *((__int64 *)&v77 + 1),
                               a3,
                               (__int64)v63,
                               v64,
                               v60,
                               v61);
    v52 = s;
    v53 = v65;
    if ( s != v101 )
      goto LABEL_34;
    goto LABEL_35;
  }
  v90 = sub_3007240((__int64)&v91);
  v19 = v74 - v90;
  v73 = v90;
  if ( !sub_3007100((__int64)&v91) )
    goto LABEL_39;
LABEL_11:
  if ( v73 )
  {
    v20 = v73;
    if ( v74 )
    {
      v21 = v74;
      for ( i = v73 % v74; i; i = v23 )
      {
        v23 = v21 % i;
        v21 = i;
      }
      v20 = v21;
    }
  }
  else
  {
    v20 = v74;
  }
  v24 = v86;
  v25 = *(__int64 **)(a1[1] + 64);
  LODWORD(s) = v20;
  BYTE4(s) = 1;
  v78 = 0;
  v87 = sub_2D43AD0(v86, v20);
  if ( !v87 )
  {
    v75 = sub_3009450(v25, v24, v83, (__int64)s, v26, v27);
    v87 = v75;
    v78 = v66;
  }
  v28 = HIWORD(v75);
  v29 = 0;
  v30 = HIWORD(v75);
  s = v101;
  v100 = 0x300000000LL;
  v84 = v73 / v20;
  if ( v73 < v20 )
  {
    v40 = 0;
  }
  else
  {
    v76 = v20;
    HIWORD(v20) = v28;
    v31 = 0;
    do
    {
      v32 = (_QWORD *)a1[1];
      *(_QWORD *)&v33 = sub_3400EE0((__int64)v32, v19, (__int64)&v93, 0, a3);
      LOWORD(v20) = v87;
      v35 = sub_3406EB0(v32, 0xA1u, (__int64)&v93, v20, v78, v34, v77, v33);
      v36 = (unsigned int)v100;
      v29 = v37;
      v38 = (unsigned int)v100 + 1LL;
      if ( v38 > HIDWORD(v100) )
      {
        v71 = v35;
        v72 = v29;
        sub_C8D5F0((__int64)&s, v101, v38, 0x10u, (__int64)v35, v29);
        v36 = (unsigned int)v100;
        v35 = v71;
        v29 = v72;
      }
      v39 = (unsigned __int8 **)((char *)s + 16 * v36);
      ++v31;
      v19 += v76;
      *v39 = v35;
      v39[1] = (unsigned __int8 *)v29;
      LODWORD(v100) = v100 + 1;
    }
    while ( v31 < v84 );
    v40 = v84;
    v30 = HIWORD(v20);
    v20 = v76;
    if ( v73 < v76 )
      v40 = 1;
  }
  if ( v40 >= v74 / v20 )
  {
    v50 = v100;
  }
  else
  {
    HIWORD(v41) = v30;
    v42 = v78;
    do
    {
      v43 = (_QWORD *)a1[1];
      LOWORD(v41) = v87;
      v97 = 0;
      v98 = 0;
      v44 = sub_33F17F0(v43, 51, (__int64)&v97, v41, v42);
      v46 = (__int64)v44;
      v29 = v45;
      if ( v97 )
      {
        v79 = v44;
        v81 = v45;
        sub_B91220((__int64)&v97, v97);
        v46 = (__int64)v79;
        v29 = v81;
      }
      v47 = (unsigned int)v100;
      v48 = (unsigned int)v100 + 1LL;
      if ( v48 > HIDWORD(v100) )
      {
        v80 = v46;
        v82 = v29;
        sub_C8D5F0((__int64)&s, v101, v48, 0x10u, v46, v29);
        v47 = (unsigned int)v100;
        v46 = v80;
        v29 = v82;
      }
      v49 = (__int64 *)((char *)s + 16 * v47);
      ++v40;
      *v49 = v46;
      v49[1] = v29;
      v50 = v100 + 1;
      LODWORD(v100) = v100 + 1;
    }
    while ( v40 != v74 / v20 );
  }
  *((_QWORD *)&v69 + 1) = v50;
  *(_QWORD *)&v69 = s;
  v51 = sub_33FC220((_QWORD *)a1[1], 159, (__int64)&v93, (unsigned int)v95, v96, v29, v69);
  v52 = s;
  v53 = v51;
  if ( s != v101 )
LABEL_34:
    _libc_free((unsigned __int64)v52);
LABEL_35:
  if ( v93 )
    sub_B91220((__int64)&v93, v93);
  return v53;
}
