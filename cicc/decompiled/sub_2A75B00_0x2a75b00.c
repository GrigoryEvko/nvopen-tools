// Function: sub_2A75B00
// Address: 0x2a75b00
//
void __fastcall sub_2A75B00(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v5; // ax
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // ecx
  __int64 v13; // r8
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r10
  char *v18; // r15
  __int64 *v19; // rax
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r13
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // r15
  __int64 *v27; // r13
  __int64 *v28; // r14
  unsigned __int64 v29; // rsi
  unsigned __int16 v30; // ax
  char v31; // r14
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // r13
  __int64 v36; // rdx
  int v37; // eax
  _QWORD *v38; // rdi
  __int64 v39; // r10
  unsigned __int64 v40; // r14
  __int64 v41; // r15
  unsigned int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdx
  int v46; // r11d
  __int64 v47; // rsi
  int v48; // r11d
  unsigned int v49; // r9d
  __int64 *v50; // rax
  __int64 v51; // rdi
  char *v52; // rdx
  __int64 *v53; // r13
  __int64 *v54; // rax
  int v55; // eax
  __int64 v56; // r14
  __int64 v57; // r15
  unsigned __int64 v58; // r13
  int v59; // eax
  __int64 v60; // r13
  __int64 v61; // rcx
  __int64 v62; // rdi
  __int64 v63; // r9
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  _QWORD *v69; // r13
  _QWORD *v70; // rax
  bool v71; // zf
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rdx
  unsigned __int64 *v76; // r15
  unsigned __int64 *v77; // rdi
  int v78; // ebx
  int v79; // r9d
  int v80; // eax
  int v81; // r8d
  __int64 v82; // [rsp-8h] [rbp-D8h]
  int v83; // [rsp+Ch] [rbp-C4h]
  __int64 v85; // [rsp+18h] [rbp-B8h]
  __int64 v86; // [rsp+18h] [rbp-B8h]
  __int64 *v87; // [rsp+20h] [rbp-B0h]
  __int64 *v88; // [rsp+28h] [rbp-A8h]
  __int64 v89; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v90; // [rsp+30h] [rbp-A0h]
  __int64 v91; // [rsp+30h] [rbp-A0h]
  char *v92; // [rsp+30h] [rbp-A0h]
  __int16 v93; // [rsp+30h] [rbp-A0h]
  __int64 v94[2]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v95; // [rsp+50h] [rbp-80h] BYREF
  __int64 v96; // [rsp+58h] [rbp-78h]
  __int64 v97; // [rsp+60h] [rbp-70h]
  char v98; // [rsp+68h] [rbp-68h]
  __int64 *v99; // [rsp+70h] [rbp-60h] BYREF
  __int64 v100; // [rsp+78h] [rbp-58h]
  _BYTE v101[80]; // [rsp+80h] [rbp-50h] BYREF

  v5 = *(_WORD *)(a2 + 2);
  v6 = (unsigned __int64)((*(_BYTE *)(a2 + 1) & 2) != 0) << 32;
  v83 = v5 & 0x3F;
  v90 = v6 | v5 & 0x3F;
  if ( a3 == *(_QWORD *)(a2 - 64) )
  {
    v8 = -32;
    v9 = -64;
    v85 = v5 & 0x3F;
  }
  else
  {
    v7 = v6 | v5 & 0x3F;
    v8 = -64;
    v9 = -32;
    v85 = (unsigned int)sub_B52F50(v5 & 0x3F);
    v90 = v85 | v7 & 0xFFFFFFFF00000000LL;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(a2 + 40);
  v12 = *(_DWORD *)(v10 + 24);
  v13 = *(_QWORD *)(v10 + 8);
  if ( v12 )
  {
    v14 = v12 - 1;
    v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( v11 == *v16 )
    {
LABEL_5:
      v18 = (char *)v16[1];
      goto LABEL_6;
    }
    v55 = 1;
    while ( v17 != -4096 )
    {
      v79 = v55 + 1;
      v15 = v14 & (v55 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v11 == *v16 )
        goto LABEL_5;
      v55 = v79;
    }
  }
  v18 = 0;
LABEL_6:
  v88 = sub_DDFBA0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 + v9), v18);
  v19 = sub_DDFBA0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 + v8), v18);
  v21 = *(_QWORD *)(a2 + 16);
  v87 = v19;
  v99 = (__int64 *)v101;
  v100 = 0x400000000LL;
  if ( v21 )
  {
    v22 = *(_QWORD *)(v21 + 24);
    v23 = (__int64 *)v101;
    v24 = 0;
    v25 = v21;
    while ( 1 )
    {
      v23[v24] = v22;
      v24 = (unsigned int)(v100 + 1);
      LODWORD(v100) = v100 + 1;
      v25 = *(_QWORD *)(v25 + 8);
      if ( !v25 )
        break;
      v22 = *(_QWORD *)(v25 + 24);
      if ( v24 + 1 > (unsigned __int64)HIDWORD(v100) )
      {
        sub_C8D5F0((__int64)&v99, v101, v24 + 1, 8u, v21, v20);
        v24 = (unsigned int)v100;
      }
      v23 = v99;
    }
    v26 = *(_QWORD *)(a1 + 24);
    v21 = 0;
    v27 = &v99[v24];
    if ( v99 != v27 )
    {
      v28 = v99 + 1;
      v29 = *v99;
      while ( v27 != v28 )
      {
        if ( v29 )
          v29 = sub_B1A110(v26, v29, *v28);
        else
          v29 = *v28;
        ++v28;
      }
      v21 = v29;
    }
  }
  v30 = sub_DDCA80(*(__int64 **)(a1 + 16), v85 | v90 & 0xFFFFFFFF00000000LL, v88, v87, v21);
  v31 = v30;
  if ( HIBYTE(v30) )
  {
    sub_DAC8D0(*(_QWORD *)(a1 + 16), (_BYTE *)a2);
    v32 = (__int64 *)sub_BD5C60(a2);
    v33 = sub_ACD760(v32, v31);
    sub_BD84D0(a2, v33);
    v35 = *(_QWORD *)(a1 + 48);
    v36 = *(unsigned int *)(v35 + 8);
    v37 = v36;
    if ( *(_DWORD *)(v35 + 12) <= (unsigned int)v36 )
    {
      v76 = (unsigned __int64 *)sub_C8D7D0(*(_QWORD *)(a1 + 48), v35 + 16, 0, 0x18u, &v95, v34);
      v77 = &v76[3 * *(unsigned int *)(v35 + 8)];
      if ( v77 )
      {
        *v77 = 6;
        v77[1] = 0;
        v77[2] = a2;
        if ( a2 != -4096 && a2 != -8192 )
          sub_BD73F0((__int64)v77);
      }
      sub_F17F80(v35, v76);
      v78 = v95;
      if ( v35 + 16 != *(_QWORD *)v35 )
        _libc_free(*(_QWORD *)v35);
      ++*(_DWORD *)(v35 + 8);
      *(_QWORD *)v35 = v76;
      *(_DWORD *)(v35 + 12) = v78;
    }
    else
    {
      v38 = (_QWORD *)(*(_QWORD *)v35 + 24 * v36);
      if ( v38 )
      {
        *v38 = 6;
        v38[1] = 0;
        v38[2] = a2;
        if ( a2 != -8192 && a2 != -4096 )
          sub_BD73F0((__int64)v38);
        v37 = *(_DWORD *)(v35 + 8);
      }
      *(_DWORD *)(v35 + 8) = v37 + 1;
    }
    goto LABEL_28;
  }
  v39 = sub_D4B130(*(_QWORD *)a1);
  if ( v39 )
  {
    v40 = ((unsigned __int64)((*(_BYTE *)(a2 + 1) & 2) != 0) << 32) | *(_WORD *)(a2 + 2) & 0x3F;
    if ( a3 == *(_QWORD *)(a2 - 64) )
    {
      v41 = -32;
      v43 = -64;
    }
    else
    {
      v91 = v39;
      v41 = -64;
      v42 = sub_B52F50(*(_WORD *)(a2 + 2) & 0x3F);
      v39 = v91;
      v43 = -32;
      v40 = v42 | v40 & 0xFFFFFFFF00000000LL;
    }
    v44 = *(_QWORD *)(a1 + 8);
    v45 = *(_QWORD *)(a2 + 40);
    v46 = *(_DWORD *)(v44 + 24);
    v47 = *(_QWORD *)(v44 + 8);
    if ( v46 )
    {
      v48 = v46 - 1;
      v49 = v48 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v50 = (__int64 *)(v47 + 16LL * v49);
      v51 = *v50;
      if ( *v50 == v45 )
      {
LABEL_37:
        v52 = (char *)v50[1];
LABEL_38:
        v86 = v39;
        v92 = v52;
        v53 = sub_DDFBA0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 + v43), v52);
        v54 = sub_DDFBA0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 + v41), v92);
        sub_DDE390((__int64)&v95, *(__int64 **)(a1 + 16), v40, (__int64)v53, (__int64)v54, *(_QWORD *)a1, a2);
        if ( v98 )
        {
          v56 = v96;
          v57 = v97;
          v93 = v95;
          v58 = *(_QWORD *)(v86 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v58 == v86 + 48 )
          {
            v60 = 0;
          }
          else
          {
            if ( !v58 )
              BUG();
            v59 = *(unsigned __int8 *)(v58 - 24);
            v60 = v58 - 24;
            if ( (unsigned int)(v59 - 30) >= 0xB )
              v60 = 0;
          }
          v61 = *(_QWORD *)a1;
          v62 = *(_QWORD *)(a1 + 40);
          v63 = *(_QWORD *)(a1 + 32);
          v94[0] = v96;
          v94[1] = v97;
          if ( !(unsigned __int8)sub_F6CE90(v62, v94, 2, v61, 2 * LODWORD(qword_4F8C268[8]), v63, v60)
            && (unsigned __int8)sub_F80650(*(__int64 **)(a1 + 40), v56, v60, v82, v64, v65)
            && (unsigned __int8)sub_F80650(*(__int64 **)(a1 + 40), v57, v60, v66, v67, v68) )
          {
            v89 = v60 + 24;
            v69 = sub_F8DB90(*(_QWORD *)(a1 + 40), v56, *(_QWORD *)(a3 + 8), v60 + 24, 0);
            v70 = sub_F8DB90(*(_QWORD *)(a1 + 40), v57, *(_QWORD *)(a3 + 8), v89, 0);
            v71 = *(_QWORD *)(a2 - 64) == 0;
            *(_WORD *)(a2 + 2) = v93 | *(_WORD *)(a2 + 2) & 0xFFC0;
            if ( !v71 )
            {
              v72 = *(_QWORD *)(a2 - 56);
              **(_QWORD **)(a2 - 48) = v72;
              if ( v72 )
                *(_QWORD *)(v72 + 16) = *(_QWORD *)(a2 - 48);
            }
            *(_QWORD *)(a2 - 64) = v69;
            if ( v69 )
            {
              v73 = v69[2];
              *(_QWORD *)(a2 - 56) = v73;
              if ( v73 )
                *(_QWORD *)(v73 + 16) = a2 - 56;
              *(_QWORD *)(a2 - 48) = v69 + 2;
              v69[2] = a2 - 64;
            }
            if ( *(_QWORD *)(a2 - 32) )
            {
              v74 = *(_QWORD *)(a2 - 24);
              **(_QWORD **)(a2 - 16) = v74;
              if ( v74 )
                *(_QWORD *)(v74 + 16) = *(_QWORD *)(a2 - 16);
            }
            *(_QWORD *)(a2 - 32) = v70;
            if ( v70 )
            {
              v75 = v70[2];
              *(_QWORD *)(a2 - 24) = v75;
              if ( v75 )
                *(_QWORD *)(v75 + 16) = a2 - 24;
              *(_QWORD *)(a2 - 16) = v70 + 2;
              v70[2] = a2 - 32;
            }
            *(_BYTE *)(a1 + 57) = 1;
            goto LABEL_28;
          }
        }
        goto LABEL_39;
      }
      v80 = 1;
      while ( v51 != -4096 )
      {
        v81 = v80 + 1;
        v49 = v48 & (v80 + v49);
        v50 = (__int64 *)(v47 + 16LL * v49);
        v51 = *v50;
        if ( v45 == *v50 )
          goto LABEL_37;
        v80 = v81;
      }
    }
    v52 = 0;
    goto LABEL_38;
  }
LABEL_39:
  if ( sub_B532B0(v83)
    && (unsigned __int8)sub_DBED40(*(_QWORD *)(a1 + 16), (__int64)v88)
    && (unsigned __int8)sub_DBED40(*(_QWORD *)(a1 + 16), (__int64)v87) )
  {
    *(_WORD *)(a2 + 2) = sub_B52EF0(v83) | *(_WORD *)(a2 + 2) & 0xFFC0;
LABEL_28:
    *(_BYTE *)(a1 + 56) = 1;
  }
  if ( v99 != (__int64 *)v101 )
    _libc_free((unsigned __int64)v99);
}
