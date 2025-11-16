// Function: sub_2AD9B30
// Address: 0x2ad9b30
//
__int64 *__fastcall sub_2AD9B30(__int64 *a1, __int64 *a2, int *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  char v13; // bl
  int v15; // r12d
  int v16; // r14d
  __int64 v17; // rax
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r15
  int v35; // r11d
  __int64 *v36; // rdi
  unsigned int v37; // ecx
  __int64 *v38; // rax
  __int64 v39; // r8
  _QWORD *v40; // rax
  __int64 v41; // rdx
  int v42; // esi
  int v43; // ecx
  __int64 v44; // rsi
  char v47; // [rsp+18h] [rbp-198h]
  __int64 v48; // [rsp+28h] [rbp-188h] BYREF
  __int64 *v49; // [rsp+30h] [rbp-180h] BYREF
  __int64 v50; // [rsp+38h] [rbp-178h]
  __int64 v51; // [rsp+40h] [rbp-170h]
  unsigned int v52; // [rsp+48h] [rbp-168h]
  _QWORD v53[4]; // [rsp+50h] [rbp-160h] BYREF
  __int64 v54; // [rsp+70h] [rbp-140h]
  __int64 v55; // [rsp+78h] [rbp-138h]
  unsigned int v56; // [rsp+80h] [rbp-130h]
  __int64 v57[9]; // [rsp+90h] [rbp-120h] BYREF
  __int64 v58; // [rsp+D8h] [rbp-D8h]
  __int64 v59; // [rsp+E0h] [rbp-D0h]
  unsigned int v60; // [rsp+E8h] [rbp-C8h]
  __int64 v61; // [rsp+F0h] [rbp-C0h]
  __int64 v62; // [rsp+F8h] [rbp-B8h]
  __int64 v63; // [rsp+100h] [rbp-B0h]
  unsigned int v64; // [rsp+108h] [rbp-A8h]
  __int64 v65; // [rsp+110h] [rbp-A0h] BYREF
  __int64 v66; // [rsp+118h] [rbp-98h]
  __int64 v67; // [rsp+120h] [rbp-90h]
  unsigned int v68; // [rsp+128h] [rbp-88h]
  _BYTE *v69; // [rsp+130h] [rbp-80h]
  __int64 v70; // [rsp+138h] [rbp-78h]
  _BYTE v71[32]; // [rsp+140h] [rbp-70h] BYREF
  __int64 v72; // [rsp+160h] [rbp-50h]
  __int64 v73; // [rsp+168h] [rbp-48h]
  __int64 v74; // [rsp+170h] [rbp-40h]
  unsigned int v75; // [rsp+178h] [rbp-38h]

  v4 = *a2;
  v5 = sub_22077B0(0x290u);
  v6 = v5;
  if ( v5 )
    sub_2BF0E10(v5, v4);
  *a1 = v6;
  v7 = a2[1];
  v8 = *a2;
  v53[2] = v6;
  v53[1] = v7;
  v53[0] = v8;
  v53[3] = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  sub_2C0CEC0(v53);
  sub_2C06CE0(*a1, *(_QWORD *)(a2[5] + 336), a2[8], 1, 0, *a2);
  v12 = *((unsigned __int8 *)a3 + 12);
  v13 = *((_BYTE *)a3 + 4);
  v15 = *a3;
  v16 = a3[2];
  v47 = *((_BYTE *)a3 + 12);
  while ( v15 != v16 || v13 != v47 )
  {
    v17 = *a1;
    LODWORD(v57[0]) = v15;
    v15 *= 2;
    BYTE4(v57[0]) = v13;
    sub_2AD9850(v17 + 48, v57, v9, v12, v10, v11);
  }
  v19 = a2[3];
  v20 = *(_QWORD *)(a2[8] + 112);
  v57[0] = (__int64)a2;
  sub_2C380F0(a1, sub_2AA7A90, v57, v20, v19);
  v21 = a2[5];
  v22 = *a1;
  v57[0] = 0;
  sub_2AC4500(v22, *(_QWORD *)(v21 + 336), 1, v57);
  sub_9C6650(v57);
  v23 = a2[8];
  v24 = a2[6];
  v25 = a2[5];
  v26 = a2[4];
  v57[7] = (__int64)(a2 + 67);
  v27 = a2[3];
  v28 = *a2;
  v57[6] = v23;
  v29 = *a1;
  v69 = v71;
  v57[1] = v28;
  v57[2] = v27;
  v57[3] = v26;
  v57[4] = v25;
  v57[5] = v24;
  v57[0] = v29;
  v70 = 0x400000000LL;
  v57[8] = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v30 = sub_2BF3F10(v29);
  v31 = sub_2BF04D0(v30);
  v32 = sub_2BF05A0(v31);
  v33 = *(_QWORD *)(v31 + 120);
  v34 = v32;
  if ( v33 != v32 )
  {
    while ( 1 )
    {
      if ( !v33 )
        BUG();
      if ( *(_BYTE *)(v33 - 16) == 29 )
        goto LABEL_12;
      v41 = *(_QWORD *)(v33 + 112);
      v42 = v68;
      v48 = v41;
      if ( !v68 )
        break;
      v35 = 1;
      v36 = 0;
      v37 = (v68 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v38 = (__int64 *)(v66 + 16LL * v37);
      v39 = *v38;
      if ( v41 != *v38 )
      {
        while ( v39 != -4096 )
        {
          if ( !v36 && v39 == -8192 )
            v36 = v38;
          v37 = (v68 - 1) & (v35 + v37);
          v38 = (__int64 *)(v66 + 16LL * v37);
          v39 = *v38;
          if ( v41 == *v38 )
            goto LABEL_10;
          ++v35;
        }
        if ( !v36 )
          v36 = v38;
        ++v65;
        v43 = v67 + 1;
        v49 = v36;
        if ( 4 * ((int)v67 + 1) < 3 * v68 )
        {
          if ( v68 - HIDWORD(v67) - v43 <= v68 >> 3 )
          {
LABEL_18:
            sub_2AC62D0((__int64)&v65, v42);
            sub_2ABE1D0((__int64)&v65, &v48, &v49);
            v41 = v48;
            v36 = v49;
            v43 = v67 + 1;
          }
          LODWORD(v67) = v43;
          if ( *v36 != -4096 )
            --HIDWORD(v67);
          *v36 = v41;
          v40 = v36 + 1;
          v36[1] = 0;
          goto LABEL_11;
        }
LABEL_17:
        v42 = 2 * v68;
        goto LABEL_18;
      }
LABEL_10:
      v40 = v38 + 1;
LABEL_11:
      *v40 = v33 - 24;
LABEL_12:
      v33 = *(_QWORD *)(v33 + 8);
      if ( v33 == v34 )
        goto LABEL_22;
    }
    ++v65;
    v49 = 0;
    goto LABEL_17;
  }
LABEL_22:
  v44 = *a1;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  sub_2AD6730((__int64)v57, v44, (__int64)&v49);
  sub_C7D6A0(v50, 16LL * v52, 8);
  sub_C7D6A0(v73, 16LL * v75, 8);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  sub_C7D6A0(v66, 16LL * v68, 8);
  sub_C7D6A0(v62, 16LL * v64, 8);
  sub_C7D6A0(v58, 24LL * v60, 8);
  sub_C7D6A0(v54, 16LL * v56, 8);
  return a1;
}
