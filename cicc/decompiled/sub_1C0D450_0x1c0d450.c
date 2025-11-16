// Function: sub_1C0D450
// Address: 0x1c0d450
//
__int64 __fastcall sub_1C0D450(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r11
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r8
  int v15; // r11d
  _QWORD *v16; // r10
  unsigned int v17; // edx
  _QWORD *v18; // rcx
  _QWORD *v19; // rdi
  unsigned int v20; // esi
  _QWORD *v21; // rax
  int v22; // ecx
  __int64 v23; // rdi
  __int64 v24; // r15
  __int64 v25; // rbx
  int v26; // r8d
  __int64 v27; // rcx
  int v28; // r9d
  __int64 v29; // rdx
  unsigned int v30; // r11d
  _QWORD *v31; // rax
  _QWORD *v32; // rdi
  _QWORD *v33; // r10
  __int64 v34; // r13
  __int64 v35; // rdi
  int v36; // eax
  int v37; // edx
  __int64 v38; // rsi
  int v39; // edi
  unsigned int v40; // eax
  _QWORD *v41; // rcx
  __int64 v42; // rax
  _QWORD *v43; // r12
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // r15
  __int64 v48; // r8
  __int64 *v49; // r10
  int v50; // r11d
  unsigned int v51; // eax
  __int64 *v52; // rdi
  __int64 v53; // rcx
  unsigned int v54; // esi
  __int64 *v55; // r14
  int v56; // edx
  _QWORD *v57; // rax
  int v58; // eax
  __int64 v59; // rdx
  __int64 *v60; // r12
  __int64 *v61; // r13
  unsigned int v62; // esi
  __int64 v63; // r8
  unsigned int v64; // eax
  __int64 *v65; // rdi
  __int64 v66; // rcx
  __int64 *v67; // r10
  int v68; // edx
  int v69; // r11d
  int v70; // eax
  int v71; // edi
  _QWORD *v72; // r14
  int v73; // r11d
  _QWORD *v74; // r9
  int v75; // eax
  int v76; // edx
  _QWORD *v77; // rcx
  _QWORD *v78; // r10
  __int64 v79; // [rsp+10h] [rbp-B0h]
  __int64 v80; // [rsp+10h] [rbp-B0h]
  __int64 v83; // [rsp+28h] [rbp-98h]
  _QWORD *v84; // [rsp+30h] [rbp-90h] BYREF
  _QWORD *v85; // [rsp+38h] [rbp-88h] BYREF
  __int64 v86[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v87; // [rsp+50h] [rbp-70h]
  __int64 v88; // [rsp+58h] [rbp-68h]
  __int64 v89; // [rsp+60h] [rbp-60h]
  __int64 v90; // [rsp+68h] [rbp-58h]
  __int64 v91; // [rsp+70h] [rbp-50h]
  __int64 v92; // [rsp+78h] [rbp-48h]
  __int64 v93; // [rsp+80h] [rbp-40h]
  __int64 *v94; // [rsp+88h] [rbp-38h]

  result = sub_1C0A150(a1, a2);
  if ( *(_DWORD *)(result + 32) )
  {
    v4 = result;
    v5 = sub_22077B0(32);
    v8 = v5;
    if ( v5 )
    {
      *(_QWORD *)v5 = 0;
      *(_QWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v5 + 16) = 0;
      *(_DWORD *)(v5 + 24) = 0;
    }
    v9 = *(unsigned int *)(a1 + 112);
    if ( (unsigned int)v9 >= *(_DWORD *)(a1 + 116) )
    {
      v83 = v8;
      sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 8, v6, v7);
      v9 = *(unsigned int *)(a1 + 112);
      v8 = v83;
    }
    v79 = v8;
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v9) = v8;
    ++*(_DWORD *)(a1 + 112);
    v10 = 0;
    *(_QWORD *)(v4 + 72) = v8;
    v86[0] = 0;
    v86[1] = 0;
    v87 = 0;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    v91 = 0;
    v92 = 0;
    v93 = 0;
    v94 = 0;
    sub_1C08D60(v86, 0);
    v11 = v79;
    v12 = *(unsigned int *)(v4 + 32);
    v13 = 8 * v12;
    if ( (_DWORD)v12 )
    {
      while ( 1 )
      {
        v20 = *(_DWORD *)(v79 + 24);
        v21 = *(_QWORD **)(*(_QWORD *)(v4 + 24) + v10);
        v84 = v21;
        if ( !v20 )
          break;
        v14 = *(_QWORD *)(v79 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v18 = (_QWORD *)(v14 + 8LL * v17);
        v19 = (_QWORD *)*v18;
        if ( v21 != (_QWORD *)*v18 )
        {
          while ( v19 != (_QWORD *)-8LL )
          {
            if ( v19 != (_QWORD *)-16LL || v16 )
              v18 = v16;
            v17 = (v20 - 1) & (v15 + v17);
            v19 = *(_QWORD **)(v14 + 8LL * v17);
            if ( v21 == v19 )
              goto LABEL_9;
            ++v15;
            v16 = v18;
            v18 = (_QWORD *)(v14 + 8LL * v17);
          }
          v71 = *(_DWORD *)(v79 + 16);
          if ( !v16 )
            v16 = v18;
          ++*(_QWORD *)v79;
          v22 = v71 + 1;
          if ( 4 * (v71 + 1) < 3 * v20 )
          {
            if ( v20 - *(_DWORD *)(v79 + 20) - v22 > v20 >> 3 )
              goto LABEL_88;
            goto LABEL_13;
          }
LABEL_12:
          v20 *= 2;
LABEL_13:
          sub_1C0C970(v79, v20);
          sub_1C09C20(v79, (__int64 *)&v84, &v85);
          v16 = v85;
          v21 = v84;
          v22 = *(_DWORD *)(v79 + 16) + 1;
LABEL_88:
          *(_DWORD *)(v79 + 16) = v22;
          if ( *v16 != -8 )
            --*(_DWORD *)(v79 + 20);
          *v16 = v21;
          v85 = (_QWORD *)*v84;
          sub_1C09F00(v86, &v85);
        }
LABEL_9:
        v10 += 8;
        if ( v13 == v10 )
        {
          v11 = v79;
          goto LABEL_15;
        }
      }
      ++*(_QWORD *)v79;
      goto LABEL_12;
    }
LABEL_15:
    v23 = v91;
    v24 = a1;
    v25 = v11;
    if ( v87 == v91 )
      return sub_1C08CE0(v86);
LABEL_24:
    if ( v92 == v23 )
    {
      v43 = *(_QWORD **)(*(v94 - 1) + 504);
      j_j___libc_free_0(v23, 512);
      v59 = *--v94 + 512;
      v92 = *v94;
      v93 = v59;
      v91 = v92 + 504;
    }
    else
    {
      v43 = *(_QWORD **)(v23 - 8);
      v91 = v23 - 8;
    }
    v44 = *(unsigned int *)(v24 + 64);
    v84 = v43;
    if ( !(_DWORD)v44 )
      goto LABEL_27;
    v26 = v44 - 1;
    v27 = *(_QWORD *)(v24 + 48);
    v28 = 1;
    LODWORD(v29) = (v44 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
    v30 = v29;
    v31 = (_QWORD *)(v27 + 16LL * (unsigned int)v29);
    v32 = (_QWORD *)*v31;
    v33 = (_QWORD *)*v31;
    if ( (_QWORD *)*v31 == v43 )
    {
      if ( v31 != (_QWORD *)(v27 + 16 * v44) )
      {
        v34 = v31[1];
        goto LABEL_20;
      }
LABEL_27:
      v34 = 0;
      goto LABEL_28;
    }
    while ( 1 )
    {
      if ( v33 == (_QWORD *)-8LL )
        goto LABEL_27;
      v30 = v26 & (v28 + v30);
      v72 = (_QWORD *)(v27 + 16LL * v30);
      v33 = (_QWORD *)*v72;
      if ( (_QWORD *)*v72 == v43 )
        break;
      ++v28;
    }
    v73 = 1;
    v74 = 0;
    if ( v72 == (_QWORD *)(v27 + 16LL * (unsigned int)v44) )
      goto LABEL_27;
    while ( v32 != (_QWORD *)-8LL )
    {
      if ( v74 || v32 != (_QWORD *)-16LL )
        v31 = v74;
      v29 = v26 & (unsigned int)(v29 + v73);
      v78 = (_QWORD *)(v27 + 16 * v29);
      v32 = (_QWORD *)*v78;
      if ( (_QWORD *)*v78 == v43 )
      {
        v34 = v78[1];
LABEL_20:
        v35 = a3;
        v36 = *(_DWORD *)(a3 + 24);
        if ( v36 )
        {
LABEL_21:
          v37 = v36 - 1;
          v38 = *(_QWORD *)(v35 + 8);
          v39 = 1;
          v40 = (v36 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v41 = *(_QWORD **)(v38 + 8LL * v40);
          if ( v41 == v43 )
          {
LABEL_22:
            v42 = *(_QWORD *)(v34 + 72);
            if ( v42 )
            {
              v60 = *(__int64 **)(v42 + 8);
              v61 = &v60[*(unsigned int *)(v42 + 24)];
              if ( *(_DWORD *)(v42 + 16) )
              {
                if ( v60 != v61 )
                {
                  while ( *v60 == -16 || *v60 == -8 )
                  {
                    if ( ++v60 == v61 )
                      goto LABEL_23;
                  }
                  if ( v60 != v61 )
                  {
                    v62 = *(_DWORD *)(v25 + 24);
                    if ( v62 )
                      goto LABEL_61;
LABEL_67:
                    ++*(_QWORD *)v25;
                    while ( 1 )
                    {
                      v62 *= 2;
LABEL_69:
                      sub_1C0C970(v25, v62);
                      sub_1C09C20(v25, v60, &v85);
                      v67 = v85;
                      v68 = *(_DWORD *)(v25 + 16) + 1;
                      while ( 1 )
                      {
                        *(_DWORD *)(v25 + 16) = v68;
                        if ( *v67 != -8 )
                          --*(_DWORD *)(v25 + 20);
                        *v67 = *v60;
                        while ( 1 )
                        {
LABEL_62:
                          if ( ++v60 == v61 )
                            goto LABEL_23;
                          if ( *v60 != -16 && *v60 != -8 )
                          {
                            if ( v60 == v61 )
                              goto LABEL_23;
                            v62 = *(_DWORD *)(v25 + 24);
                            if ( !v62 )
                              goto LABEL_67;
LABEL_61:
                            v63 = *(_QWORD *)(v25 + 8);
                            v64 = (v62 - 1) & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
                            v65 = (__int64 *)(v63 + 8LL * v64);
                            v66 = *v65;
                            if ( *v65 != *v60 )
                              break;
                          }
                        }
                        v69 = 1;
                        v67 = 0;
                        while ( v66 != -8 )
                        {
                          if ( v67 || v66 != -16 )
                            v65 = v67;
                          v64 = (v62 - 1) & (v69 + v64);
                          v66 = *(_QWORD *)(v63 + 8LL * v64);
                          if ( *v60 == v66 )
                            goto LABEL_62;
                          ++v69;
                          v67 = v65;
                          v65 = (__int64 *)(v63 + 8LL * v64);
                        }
                        v70 = *(_DWORD *)(v25 + 16);
                        if ( !v67 )
                          v67 = v65;
                        ++*(_QWORD *)v25;
                        v68 = v70 + 1;
                        if ( 4 * (v70 + 1) >= 3 * v62 )
                          break;
                        if ( v62 - *(_DWORD *)(v25 + 20) - v68 <= v62 >> 3 )
                          goto LABEL_69;
                      }
                    }
                  }
                }
              }
            }
LABEL_23:
            v23 = v91;
            if ( v91 == v87 )
              return sub_1C08CE0(v86);
            goto LABEL_24;
          }
          while ( v41 != (_QWORD *)-8LL )
          {
            v40 = v37 & (v39 + v40);
            v41 = *(_QWORD **)(v38 + 8LL * v40);
            if ( v41 == v43 )
              goto LABEL_22;
            ++v39;
          }
        }
LABEL_29:
        v45 = *(unsigned int *)(v34 + 32);
        if ( !(_DWORD)v45 )
          goto LABEL_23;
        v80 = v24;
        v46 = 0;
        v47 = 8 * v45;
        while ( 2 )
        {
          while ( 2 )
          {
            v54 = *(_DWORD *)(v25 + 24);
            v55 = (__int64 *)(v46 + *(_QWORD *)(v34 + 24));
            if ( !v54 )
            {
              ++*(_QWORD *)v25;
              goto LABEL_35;
            }
            v48 = *(_QWORD *)(v25 + 8);
            v49 = 0;
            v50 = 1;
            v51 = (v54 - 1) & (((unsigned int)*v55 >> 9) ^ ((unsigned int)*v55 >> 4));
            v52 = (__int64 *)(v48 + 8LL * v51);
            v53 = *v52;
            if ( *v55 == *v52 )
            {
LABEL_32:
              v46 += 8;
              if ( v46 == v47 )
                goto LABEL_40;
              continue;
            }
            break;
          }
          while ( v53 != -8 )
          {
            if ( v49 || v53 != -16 )
              v52 = v49;
            v51 = (v54 - 1) & (v50 + v51);
            v53 = *(_QWORD *)(v48 + 8LL * v51);
            if ( *v55 == v53 )
              goto LABEL_32;
            ++v50;
            v49 = v52;
            v52 = (__int64 *)(v48 + 8LL * v51);
          }
          v58 = *(_DWORD *)(v25 + 16);
          if ( !v49 )
            v49 = v52;
          ++*(_QWORD *)v25;
          v56 = v58 + 1;
          if ( 4 * (v58 + 1) < 3 * v54 )
          {
            if ( v54 - *(_DWORD *)(v25 + 20) - v56 <= v54 >> 3 )
            {
LABEL_36:
              sub_1C0C970(v25, v54);
              sub_1C09C20(v25, v55, &v85);
              v49 = v85;
              v56 = *(_DWORD *)(v25 + 16) + 1;
            }
            *(_DWORD *)(v25 + 16) = v56;
            if ( *v49 != -8 )
              --*(_DWORD *)(v25 + 20);
            *v49 = *v55;
            v57 = *(_QWORD **)(*(_QWORD *)(v34 + 24) + v46);
            v46 += 8;
            v85 = (_QWORD *)*v57;
            sub_1C09F00(v86, &v85);
            if ( v46 == v47 )
            {
LABEL_40:
              v24 = v80;
              v23 = v91;
              if ( v91 == v87 )
                return sub_1C08CE0(v86);
              goto LABEL_24;
            }
            continue;
          }
          break;
        }
LABEL_35:
        v54 *= 2;
        goto LABEL_36;
      }
      ++v73;
      v74 = v31;
      v31 = (_QWORD *)(v27 + 16 * v29);
    }
    if ( !v74 )
      v74 = v31;
    v75 = *(_DWORD *)(v24 + 56);
    ++*(_QWORD *)(v24 + 40);
    v76 = v75 + 1;
    if ( 4 * (v75 + 1) >= (unsigned int)(3 * v44) )
    {
      LODWORD(v44) = 2 * v44;
    }
    else
    {
      v77 = v43;
      if ( (int)v44 - *(_DWORD *)(v24 + 60) - v76 > (unsigned int)v44 >> 3 )
      {
LABEL_100:
        *(_DWORD *)(v24 + 56) = v76;
        if ( *v74 != -8 )
          --*(_DWORD *)(v24 + 60);
        *v74 = v77;
        v34 = 0;
        v74[1] = 0;
LABEL_28:
        v35 = a3;
        v36 = *(_DWORD *)(a3 + 24);
        if ( v36 )
          goto LABEL_21;
        goto LABEL_29;
      }
    }
    sub_1C04E30(v24 + 40, v44);
    sub_1C09800(v24 + 40, (__int64 *)&v84, &v85);
    v74 = v85;
    v77 = v84;
    v76 = *(_DWORD *)(v24 + 56) + 1;
    goto LABEL_100;
  }
  return result;
}
