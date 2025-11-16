// Function: sub_2400A70
// Address: 0x2400a70
//
__int64 __fastcall sub_2400A70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // r11
  unsigned __int64 v6; // r15
  __int64 *v7; // rbx
  __int64 *v8; // r14
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  unsigned int v13; // eax
  _QWORD *v14; // r9
  __int64 v15; // rsi
  int v16; // edx
  _QWORD *v17; // rax
  __int64 *v18; // r10
  __int64 *v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rsi
  int v27; // eax
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // r13
  __int64 v31; // r8
  __int64 v32; // rcx
  int v33; // r9d
  __int64 *v34; // rdx
  __int64 v35; // r11
  __int64 *v36; // rax
  _QWORD *v37; // rdi
  __int64 *v38; // rbx
  __int64 v39; // rax
  void *v40; // rax
  __int64 v41; // rdx
  void *v42; // r12
  __int64 v43; // rsi
  __int64 *v44; // rbx
  __int64 *v45; // r12
  __int64 v46; // rsi
  int v48; // r10d
  int v49; // r8d
  unsigned int v50; // r13d
  _QWORD *v51; // rdi
  __int64 v52; // rcx
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // r10
  __int64 v56; // rdi
  unsigned __int64 v57; // r13
  int v58; // eax
  int v59; // edi
  int v60; // r10d
  _QWORD *v61; // r8
  __int64 v63; // [rsp+10h] [rbp-D0h]
  __int64 v64; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v67; // [rsp+28h] [rbp-B8h]
  __int64 v68; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v69; // [rsp+38h] [rbp-A8h]
  __int64 *v70; // [rsp+38h] [rbp-A8h]
  _QWORD *v71; // [rsp+48h] [rbp-98h] BYREF
  __int64 v72; // [rsp+50h] [rbp-90h] BYREF
  __int64 v73; // [rsp+58h] [rbp-88h]
  __int64 v74; // [rsp+60h] [rbp-80h]
  __int64 v75; // [rsp+68h] [rbp-78h]
  __int64 v76; // [rsp+70h] [rbp-70h] BYREF
  void *src; // [rsp+78h] [rbp-68h]
  __int64 v78; // [rsp+80h] [rbp-60h]
  __int64 v79; // [rsp+88h] [rbp-58h]
  __int64 *v80; // [rsp+90h] [rbp-50h] BYREF
  __int64 v81; // [rsp+98h] [rbp-48h]
  __int64 v82; // [rsp+A0h] [rbp-40h]
  unsigned int v83; // [rsp+A8h] [rbp-38h]

  v3 = a1;
  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(_QWORD *)a2;
  v74 = 0;
  v75 = 0;
  v72 = 0;
  v73 = 0;
  v69 = v5 + 96 * v4;
  if ( v5 == v69 )
    goto LABEL_33;
  v6 = v5;
  do
  {
    v7 = *(__int64 **)(v6 + 16);
    v8 = &v7[*(unsigned int *)(v6 + 24)];
    if ( v7 != v8 )
    {
      while ( 1 )
      {
        v12 = *v7;
        if ( !(_DWORD)v75 )
          break;
        v9 = (v75 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v10 = (_QWORD *)(v73 + 8LL * v9);
        v11 = *v10;
        if ( v12 == *v10 )
        {
LABEL_6:
          if ( v8 == ++v7 )
            goto LABEL_14;
        }
        else
        {
          v48 = 1;
          v14 = 0;
          while ( v11 != -4096 )
          {
            if ( v14 || v11 != -8192 )
              v10 = v14;
            v9 = (v75 - 1) & (v48 + v9);
            v11 = *(_QWORD *)(v73 + 8LL * v9);
            if ( v12 == v11 )
              goto LABEL_6;
            ++v48;
            v14 = v10;
            v10 = (_QWORD *)(v73 + 8LL * v9);
          }
          if ( !v14 )
            v14 = v10;
          ++v72;
          v16 = v74 + 1;
          if ( 4 * ((int)v74 + 1) < (unsigned int)(3 * v75) )
          {
            if ( (int)v75 - HIDWORD(v74) - v16 <= (unsigned int)v75 >> 3 )
            {
              sub_CF4090((__int64)&v72, v75);
              if ( !(_DWORD)v75 )
              {
LABEL_98:
                LODWORD(v74) = v74 + 1;
                BUG();
              }
              v49 = 1;
              v50 = (v75 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
              v14 = (_QWORD *)(v73 + 8LL * v50);
              v16 = v74 + 1;
              v51 = 0;
              v52 = *v14;
              if ( v12 != *v14 )
              {
                while ( v52 != -4096 )
                {
                  if ( v52 == -8192 && !v51 )
                    v51 = v14;
                  v50 = (v75 - 1) & (v49 + v50);
                  v14 = (_QWORD *)(v73 + 8LL * v50);
                  v52 = *v14;
                  if ( v12 == *v14 )
                    goto LABEL_11;
                  ++v49;
                }
                if ( v51 )
                  v14 = v51;
              }
            }
            goto LABEL_11;
          }
LABEL_9:
          sub_CF4090((__int64)&v72, 2 * v75);
          if ( !(_DWORD)v75 )
            goto LABEL_98;
          v13 = (v75 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v14 = (_QWORD *)(v73 + 8LL * v13);
          v15 = *v14;
          v16 = v74 + 1;
          if ( v12 != *v14 )
          {
            v60 = 1;
            v61 = 0;
            while ( v15 != -4096 )
            {
              if ( !v61 && v15 == -8192 )
                v61 = v14;
              v13 = (v75 - 1) & (v60 + v13);
              v14 = (_QWORD *)(v73 + 8LL * v13);
              v15 = *v14;
              if ( v12 == *v14 )
                goto LABEL_11;
              ++v60;
            }
            if ( v61 )
              v14 = v61;
          }
LABEL_11:
          LODWORD(v74) = v16;
          if ( *v14 != -4096 )
            --HIDWORD(v74);
          ++v7;
          *v14 = v12;
          if ( v8 == v7 )
            goto LABEL_14;
        }
      }
      ++v72;
      goto LABEL_9;
    }
LABEL_14:
    v6 += 96LL;
  }
  while ( v6 != v69 );
  v3 = a1;
  v68 = *(_QWORD *)(a3 + 864);
  v63 = *(_QWORD *)a2 + 96LL * *(unsigned int *)(a2 + 8);
  if ( v63 != *(_QWORD *)a2 )
  {
    v67 = *(_QWORD *)a2;
    while ( 1 )
    {
      v17 = *(_QWORD **)v67;
      v76 = 0;
      src = 0;
      v71 = v17;
      v78 = 0;
      v79 = 0;
      v64 = v67;
      if ( !*(_BYTE *)(v67 + 8) )
        break;
      v53 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
      v54 = *(_QWORD *)(v53 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v54 == v53 + 48 )
        goto LABEL_97;
      if ( !v54 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 > 0xA )
      {
LABEL_97:
        v80 = 0;
        v81 = 0;
        v82 = 0;
        v83 = 0;
        BUG();
      }
      v55 = *(_QWORD *)(v3 + 16);
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v83 = 0;
      sub_24005F0(*(unsigned __int8 **)(v54 - 120), v68, v55, (__int64)&v72, (__int64)&v76, (__int64)&v80);
      sub_C7D6A0(v81, 16LL * v83, 8);
      v18 = *(__int64 **)(v67 + 16);
      v70 = &v18[*(unsigned int *)(v67 + 24)];
      if ( v18 != v70 )
        goto LABEL_19;
LABEL_21:
      v24 = *(unsigned int *)(a3 + 944);
      v25 = *(_QWORD *)(a3 + 936);
      v26 = v24 + 1;
      v27 = *(_DWORD *)(a3 + 944);
      if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 948) )
      {
        v56 = a3 + 936;
        if ( v25 > v67 || v25 + 96 * v24 <= v67 )
        {
          sub_23FAFC0(v56, v26, v25, v24, v22, v23);
          v24 = *(unsigned int *)(a3 + 944);
          v25 = *(_QWORD *)(a3 + 936);
          v27 = *(_DWORD *)(a3 + 944);
        }
        else
        {
          v57 = v67 - v25;
          sub_23FAFC0(v56, v26, v25, v24, v22, v23);
          v25 = *(_QWORD *)(a3 + 936);
          v24 = *(unsigned int *)(a3 + 944);
          v64 = v25 + v57;
          v27 = *(_DWORD *)(a3 + 944);
        }
      }
      v28 = 96 * v24 + v25;
      if ( v28 )
      {
        *(_QWORD *)v28 = *(_QWORD *)v64;
        *(_BYTE *)(v28 + 8) = *(_BYTE *)(v64 + 8);
        *(_QWORD *)(v28 + 16) = v28 + 32;
        *(_QWORD *)(v28 + 24) = 0x800000000LL;
        if ( *(_DWORD *)(v64 + 24) )
          sub_23FAD70(v28 + 16, v64 + 16, v28, v64, v22, v23);
        v27 = *(_DWORD *)(a3 + 944);
      }
      v29 = *(_DWORD *)(a3 + 1808);
      *(_DWORD *)(a3 + 944) = v27 + 1;
      v30 = a3 + 1784;
      if ( !v29 )
      {
        v80 = 0;
        ++*(_QWORD *)(a3 + 1784);
LABEL_78:
        v29 *= 2;
LABEL_79:
        sub_23FEFD0(v30, v29);
        sub_23FDB10(v30, (__int64 *)&v71, &v80);
        v32 = (__int64)v71;
        v34 = v80;
        v59 = *(_DWORD *)(a3 + 1800) + 1;
        goto LABEL_74;
      }
      v31 = *(_QWORD *)(a3 + 1792);
      v32 = (__int64)v71;
      v33 = 1;
      v34 = 0;
      LODWORD(v35) = (v29 - 1) & (((unsigned int)v71 >> 4) ^ ((unsigned int)v71 >> 9));
      v36 = (__int64 *)(v31 + 40LL * (unsigned int)v35);
      v37 = (_QWORD *)*v36;
      if ( (_QWORD *)*v36 == v71 )
      {
LABEL_28:
        v38 = v36 + 1;
        goto LABEL_29;
      }
      while ( v37 != (_QWORD *)-4096LL )
      {
        if ( !v34 && v37 == (_QWORD *)-8192LL )
          v34 = v36;
        v35 = (v29 - 1) & ((_DWORD)v35 + v33);
        v36 = (__int64 *)(v31 + 40 * v35);
        v37 = (_QWORD *)*v36;
        if ( v71 == (_QWORD *)*v36 )
          goto LABEL_28;
        ++v33;
      }
      if ( !v34 )
        v34 = v36;
      ++*(_QWORD *)(a3 + 1784);
      v58 = *(_DWORD *)(a3 + 1800);
      v80 = v34;
      v59 = v58 + 1;
      if ( 4 * (v58 + 1) >= 3 * v29 )
        goto LABEL_78;
      if ( v29 - *(_DWORD *)(a3 + 1804) - v59 <= v29 >> 3 )
        goto LABEL_79;
LABEL_74:
      *(_DWORD *)(a3 + 1800) = v59;
      if ( *v34 != -4096 )
        --*(_DWORD *)(a3 + 1804);
      *v34 = v32;
      v38 = v34 + 1;
      v34[1] = 0;
      v34[2] = 0;
      v34[3] = 0;
      *((_DWORD *)v34 + 8) = 0;
LABEL_29:
      if ( v38 == &v76 )
        goto LABEL_54;
      sub_C7D6A0(v38[1], 8LL * *((unsigned int *)v38 + 6), 8);
      v39 = (unsigned int)v79;
      *((_DWORD *)v38 + 6) = v79;
      if ( !(_DWORD)v39 )
      {
        v38[1] = 0;
        v38[2] = 0;
LABEL_54:
        v42 = src;
        v43 = 8LL * (unsigned int)v79;
        goto LABEL_32;
      }
      v40 = (void *)sub_C7D670(8 * v39, 8);
      v41 = *((unsigned int *)v38 + 6);
      v38[1] = (__int64)v40;
      v42 = src;
      v38[2] = v78;
      memcpy(v40, v42, 8 * v41);
      v43 = 8LL * (unsigned int)v79;
LABEL_32:
      sub_C7D6A0((__int64)v42, v43, 8);
      v67 += 96LL;
      if ( v63 == v67 )
        goto LABEL_33;
    }
    v18 = *(__int64 **)(v67 + 16);
    v70 = &v18[*(unsigned int *)(v67 + 24)];
    if ( v70 == v18 )
    {
      v42 = 0;
      v43 = 0;
      goto LABEL_32;
    }
LABEL_19:
    v19 = v18;
    do
    {
      v20 = *v19;
      v80 = 0;
      v21 = *(_QWORD *)(v3 + 16);
      ++v19;
      v81 = 0;
      v82 = 0;
      v83 = 0;
      sub_24005F0(*(unsigned __int8 **)(v20 - 96), v68, v21, (__int64)&v72, (__int64)&v76, (__int64)&v80);
      sub_C7D6A0(v81, 16LL * v83, 8);
    }
    while ( v70 != v19 );
    goto LABEL_21;
  }
LABEL_33:
  v44 = *(__int64 **)(a2 + 784);
  v45 = &v44[*(unsigned int *)(a2 + 792)];
  while ( v45 != v44 )
  {
    v46 = *v44++;
    sub_2400A70(v3, v46, a3);
  }
  return sub_C7D6A0(v73, 8LL * (unsigned int)v75, 8);
}
