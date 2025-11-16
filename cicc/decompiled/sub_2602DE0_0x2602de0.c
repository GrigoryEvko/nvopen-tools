// Function: sub_2602DE0
// Address: 0x2602de0
//
__int64 __fastcall sub_2602DE0(_QWORD **a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  _QWORD *v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // r15
  unsigned int v15; // esi
  __int64 v16; // rcx
  unsigned int v17; // edx
  __int64 v18; // r8
  __int64 v19; // r14
  __int64 v20; // r13
  unsigned __int64 v21; // rdi
  int v22; // eax
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rbx
  unsigned int v28; // r13d
  __int64 v29; // r12
  unsigned int v30; // ebx
  __int64 v31; // r13
  __int64 v32; // r8
  __int64 v33; // rdi
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 *v36; // rdx
  __int64 v37; // r10
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rdi
  int v42; // eax
  unsigned __int8 *v43; // rdi
  int v44; // edx
  int v45; // ecx
  int v46; // r9d
  __int64 v47; // rax
  __int64 v48; // r15
  char *v49; // rax
  size_t v50; // rdx
  __int64 *v51; // r15
  __int64 *v52; // rax
  __int64 *v53; // r14
  __int64 v54; // r9
  __int64 v55; // rbx
  __int64 v56; // r12
  unsigned __int64 v57; // rax
  int v58; // edx
  _QWORD *v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // r9
  unsigned __int64 v62; // r11
  int v63; // edx
  _QWORD *v64; // r11
  __int64 v65; // rdi
  __int128 v66; // [rsp-30h] [rbp-110h]
  __int64 v68; // [rsp+10h] [rbp-D0h]
  __int64 v69; // [rsp+10h] [rbp-D0h]
  __int64 *v70; // [rsp+18h] [rbp-C8h]
  _QWORD *v71; // [rsp+18h] [rbp-C8h]
  __int64 v72; // [rsp+18h] [rbp-C8h]
  __int64 v73; // [rsp+20h] [rbp-C0h]
  __int64 v75; // [rsp+28h] [rbp-B8h]
  __int64 v77; // [rsp+30h] [rbp-B0h]
  _QWORD *v78; // [rsp+38h] [rbp-A8h]
  __int64 v79; // [rsp+38h] [rbp-A8h]
  _QWORD *v80; // [rsp+38h] [rbp-A8h]
  __int64 v82; // [rsp+48h] [rbp-98h]
  unsigned __int16 v83; // [rsp+48h] [rbp-98h]
  __int64 v84; // [rsp+48h] [rbp-98h]
  int v85; // [rsp+48h] [rbp-98h]
  __int64 v86; // [rsp+48h] [rbp-98h]
  __int64 v87; // [rsp+48h] [rbp-98h]
  __int64 *v88; // [rsp+50h] [rbp-90h]
  __int64 v89; // [rsp+58h] [rbp-88h]
  __int64 v90; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v91; // [rsp+68h] [rbp-78h]
  __int64 v92; // [rsp+70h] [rbp-70h]
  unsigned int v93; // [rsp+78h] [rbp-68h]
  const char *v94; // [rsp+80h] [rbp-60h] BYREF
  __int64 v95; // [rsp+88h] [rbp-58h]
  __int128 v96; // [rsp+90h] [rbp-50h]
  __int64 v97; // [rsp+A0h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 152) > 1u )
  {
    v10 = *(_QWORD *)(a2 + 56);
    LOWORD(v97) = 259;
    *((_QWORD *)&v66 + 1) = v95;
    *(_QWORD *)&v66 = "final_block";
    v75 = v10;
    v90 = 0;
    v91 = 0;
    v92 = 0;
    v93 = 0;
    v94 = "final_block";
    sub_26028B0(a2 + 72, (__int64)&v90, v10, a2, a5, a6, v66, v96, v97);
    v11 = 16LL * v93;
    if ( !(_DWORD)v92 )
      return sub_C7D6A0((__int64)v91, v11, 8);
    v78 = &v91[(unsigned __int64)v11 / 8];
    if ( v91 == &v91[(unsigned __int64)v11 / 8] )
      return sub_C7D6A0((__int64)v91, v11, 8);
    v12 = v91;
    while ( 1 )
    {
      v13 = *v12;
      v14 = v12;
      if ( *v12 != -4096 && v13 != -8192 )
        break;
      v12 += 2;
      if ( &v91[(unsigned __int64)v11 / 8] == v12 )
        return sub_C7D6A0((__int64)v91, v11, 8);
    }
    if ( v78 == v12 )
      return sub_C7D6A0((__int64)v91, v11, 8);
    while ( 1 )
    {
      v15 = *(_DWORD *)(a2 + 96);
      v16 = *(_QWORD *)(a2 + 80);
      if ( !v15 )
        goto LABEL_49;
      v17 = (v15 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v18 = *(_QWORD *)(v16 + 16LL * v17);
      v88 = (__int64 *)(v16 + 16LL * v17);
      if ( v13 != v18 )
        break;
LABEL_15:
      v19 = v14[1];
      v20 = v88[1];
      v21 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 == v20 + 48 )
      {
        v23 = 0;
      }
      else
      {
        if ( !v21 )
LABEL_80:
          BUG();
        v22 = *(unsigned __int8 *)(v21 - 24);
        v23 = (_QWORD *)(v21 - 24);
        if ( (unsigned int)(v22 - 30) >= 0xB )
          v23 = 0;
      }
      v24 = v73;
      LOWORD(v24) = 0;
      v73 = v24;
      sub_B44550(v23, v14[1], (unsigned __int64 *)(v19 + 48), v24);
      v25 = v20;
      sub_B43C20((__int64)&v94, v20);
      v27 = (a4[1] - *a4) >> 5;
      v82 = *(_QWORD *)(v75 + 104);
      v28 = v82 - 1;
      if ( (*(_BYTE *)(v75 + 2) & 1) != 0 )
        sub_B2C6D0(v75, v25, v26, (unsigned int)v82);
      v70 = (__int64 *)v94;
      v83 = v95;
      v29 = *(_QWORD *)(v75 + 96) + 40LL * v28;
      v68 = sub_BD2DA0(80);
      if ( v68 )
        sub_B53A60(v68, v29, v19, v27, (__int64)v70, v83);
      v30 = 0;
      v84 = a4[1];
      if ( *a4 != v84 )
      {
        v71 = v14;
        v31 = *a4;
        do
        {
          v32 = *(_QWORD *)(v31 + 8);
          v33 = *v88;
          v34 = *(unsigned int *)(v31 + 24);
          if ( (_DWORD)v34 )
          {
            v35 = (v34 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
            v36 = (__int64 *)(v32 + 16LL * v35);
            v37 = *v36;
            if ( v33 == *v36 )
            {
LABEL_27:
              if ( v36 != (__int64 *)(v32 + 16 * v34) )
              {
                v38 = v36[1];
                v39 = sub_BCB2D0(*a1);
                v40 = sub_ACD640(v39, v30, 0);
                sub_B53E30(v68, v40, v38);
                v41 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v41 == v38 + 48 )
                {
                  v43 = 0;
                }
                else
                {
                  if ( !v41 )
                    goto LABEL_80;
                  v42 = *(unsigned __int8 *)(v41 - 24);
                  v43 = (unsigned __int8 *)(v41 - 24);
                  if ( (unsigned int)(v42 - 30) >= 0xB )
                    v43 = 0;
                }
                ++v30;
                sub_B46F90(v43, 0, v19);
              }
            }
            else
            {
              v44 = 1;
              while ( v37 != -4096 )
              {
                v45 = v44 + 1;
                v35 = (v34 - 1) & (v44 + v35);
                v36 = (__int64 *)(v32 + 16LL * v35);
                v37 = *v36;
                if ( v33 == *v36 )
                  goto LABEL_27;
                v44 = v45;
              }
            }
          }
          v31 += 32;
        }
        while ( v84 != v31 );
        v14 = v71;
      }
      v14 += 2;
      if ( v14 != v78 )
      {
        while ( 1 )
        {
          v13 = *v14;
          if ( *v14 != -4096 && v13 != -8192 )
            break;
          v14 += 2;
          if ( v78 == v14 )
            return sub_C7D6A0((__int64)v91, 16LL * v93, 8);
        }
        if ( v78 != v14 )
          continue;
      }
      return sub_C7D6A0((__int64)v91, 16LL * v93, 8);
    }
    v46 = 1;
    while ( v18 != -4096 )
    {
      v17 = (v15 - 1) & (v46 + v17);
      v18 = *(_QWORD *)(v16 + 16LL * v17);
      v88 = (__int64 *)(v16 + 16LL * v17);
      if ( v18 == v13 )
        goto LABEL_15;
      ++v46;
    }
LABEL_49:
    v88 = (__int64 *)(v16 + 16LL * v15);
    goto LABEL_15;
  }
  v8 = *a4;
  result = a4[1] - *a4;
  if ( result == 32 )
  {
    sub_C7D6A0(0, 0, 8);
    v47 = *(unsigned int *)(v8 + 24);
    if ( (_DWORD)v47 )
    {
      v48 = 16 * v47;
      v72 = 16 * v47;
      v49 = (char *)sub_C7D670(16 * v47, 8);
      v50 = v48;
      v69 = (__int64)v49;
      v85 = *(_DWORD *)(v8 + 16);
      v51 = (__int64 *)&v49[v48];
      v52 = (__int64 *)memcpy(v49, *(const void **)(v8 + 8), v50);
      if ( v85 )
      {
        while ( 1 )
        {
          v53 = v52;
          if ( *v52 != -8192 && *v52 != -4096 )
            break;
          v52 += 2;
          if ( v51 == v52 )
            return sub_C7D6A0(v69, v72, 8);
        }
        if ( v52 != v51 )
        {
          v54 = v6;
          v55 = v7;
          do
          {
            v86 = v54;
            sub_25FCDB0((__int64 **)&v94, a3, *v53);
            v56 = v53[1];
            v57 = *(_QWORD *)(v56 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v56 + 48 == v57 )
            {
              v59 = 0;
            }
            else
            {
              if ( !v57 )
                goto LABEL_80;
              v58 = *(unsigned __int8 *)(v57 - 24);
              v59 = 0;
              v60 = (_QWORD *)(v57 - 24);
              if ( (unsigned int)(v58 - 30) < 0xB )
                v59 = v60;
            }
            v79 = v86;
            v87 = *(_QWORD *)(v96 + 8);
            sub_B43D60(v59);
            v61 = v79;
            v62 = *(_QWORD *)(v87 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v87 + 48 == v62 )
            {
              v64 = 0;
            }
            else
            {
              if ( !v62 )
                goto LABEL_80;
              v63 = *(unsigned __int8 *)(v62 - 24);
              v64 = (_QWORD *)(v62 - 24);
              if ( (unsigned int)(v63 - 30) >= 0xB )
                v64 = 0;
            }
            v65 = v89;
            LOWORD(v55) = 0;
            LOWORD(v61) = 1;
            LOWORD(v65) = 0;
            v77 = v61;
            v53 += 2;
            v89 = v65;
            v80 = v64;
            sub_AA80F0(
              v87,
              (unsigned __int64 *)(v87 + 48),
              0,
              v56,
              *(__int64 **)(v56 + 56),
              v61,
              (__int64 *)(v56 + 48),
              v65);
            sub_B44550(v80, v87, (unsigned __int64 *)(v87 + 48), v55);
            sub_AA5450((_QWORD *)v56);
            if ( v53 == v51 )
              break;
            v54 = v77;
            while ( *v53 == -8192 || *v53 == -4096 )
            {
              v53 += 2;
              if ( v51 == v53 )
                return sub_C7D6A0(v69, v72, 8);
            }
          }
          while ( v53 != v51 );
        }
      }
      return sub_C7D6A0(v69, v72, 8);
    }
    else
    {
      return sub_C7D6A0(0, 0, 8);
    }
  }
  return result;
}
