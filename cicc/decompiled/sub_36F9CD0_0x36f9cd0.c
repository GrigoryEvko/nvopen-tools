// Function: sub_36F9CD0
// Address: 0x36f9cd0
//
__int64 __fastcall sub_36F9CD0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rdi
  __int64 (*v4)(void); // rax
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rsi
  __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 *v21; // r12
  unsigned int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // rdx
  int v27; // r8d
  __int64 *v28; // rdi
  int v29; // edx
  __int64 v30; // r14
  int v31; // r8d
  __int64 *v32; // rcx
  __int64 v33; // rdx
  int v34; // r10d
  __int64 *v35; // rdi
  int v36; // edx
  __int64 v37; // r14
  __int64 v38; // r13
  unsigned int v39; // eax
  __int64 v40; // r8
  int v41; // eax
  __int64 v42; // r11
  int v43; // esi
  __int64 *v44; // rcx
  __int64 *v45; // r8
  int v46; // eax
  int v47; // ecx
  __int64 v48; // rsi
  unsigned __int8 *v49; // r8
  unsigned __int8 *v50; // r13
  unsigned __int8 *v51; // r15
  __int64 *v52; // r14
  unsigned int v53; // r12d
  unsigned __int8 v54; // al
  unsigned __int8 v55; // al
  __int64 v56; // rdi
  unsigned __int8 v57; // al
  __int64 v58; // rdx
  __int64 v59; // rcx
  int v60; // r8d
  int v61; // r8d
  __int64 v62; // r9
  unsigned int v63; // ecx
  __int64 v64; // r11
  int v65; // edi
  __int64 *v66; // rsi
  int v67; // r8d
  int v68; // r8d
  __int64 v69; // r9
  __int64 *v70; // rdi
  unsigned int v71; // ebx
  int v72; // ecx
  __int64 v73; // rsi
  __int64 v74; // [rsp+10h] [rbp-90h]
  __int64 *v75; // [rsp+18h] [rbp-88h]
  __int64 v76; // [rsp+20h] [rbp-80h]
  __int64 *v78; // [rsp+30h] [rbp-70h]
  __int64 v79; // [rsp+30h] [rbp-70h]
  __int64 v80; // [rsp+38h] [rbp-68h]
  __int64 v81; // [rsp+38h] [rbp-68h]
  unsigned int v82; // [rsp+38h] [rbp-68h]
  __int64 *v83; // [rsp+40h] [rbp-60h]
  unsigned int v84; // [rsp+40h] [rbp-60h]
  unsigned __int8 v85; // [rsp+4Bh] [rbp-55h]
  char v86; // [rsp+4Ch] [rbp-54h]
  unsigned int v87; // [rsp+4Ch] [rbp-54h]
  __int64 v88; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v89; // [rsp+58h] [rbp-48h]
  __int64 v90; // [rsp+60h] [rbp-40h]
  __int64 v91; // [rsp+68h] [rbp-38h]

  v2 = a2;
  v74 = a1 + 200;
  sub_2E476F0(a1 + 200);
  v3 = a2[2];
  v76 = a2[4];
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 128LL);
  if ( (char *)v4 == (char *)sub_30594F0 )
    v5 = v3 + 376;
  else
    v5 = v4();
  v6 = *a2;
  v85 = sub_CE9220(*a2);
  if ( v85 )
  {
    if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v6, (__int64)a2, v7, v8);
      v49 = *(unsigned __int8 **)(v6 + 96);
      v50 = &v49[40 * *(_QWORD *)(v6 + 104)];
      if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v6, (__int64)a2, v58, v59);
        v49 = *(unsigned __int8 **)(v6 + 96);
      }
    }
    else
    {
      v49 = *(unsigned __int8 **)(v6 + 96);
      v50 = &v49[40 * *(_QWORD *)(v6 + 104)];
    }
    if ( v49 == v50 )
    {
      v85 = 0;
    }
    else
    {
      v79 = v5;
      v51 = v49;
      v52 = a2;
      v53 = 0;
      v82 = 0;
      v84 = 0;
      v87 = 0;
      v85 = 0;
      do
      {
        if ( (unsigned __int8)sub_CE8A80(v51, (__int64)a2) )
        {
          if ( (unsigned __int8)sub_CE8980(v51, (__int64)a2) || (unsigned __int8)sub_CE8A00(v51, (__int64)a2) )
          {
            if ( byte_50411E8 != 1 && v84 <= 0xF )
            {
              a2 = (__int64 *)v53;
              v57 = sub_36F9790(v52, v53, 2u, v74);
              if ( v57 )
              {
                ++v84;
                v85 = v57;
              }
            }
          }
          else if ( byte_50412C8 != 1 && v87 <= 0xFF )
          {
            a2 = (__int64 *)v53;
            v55 = sub_36F9790(v52, v53, 1u, v74);
            if ( v55 )
            {
              ++v87;
              v85 = v55;
            }
          }
        }
        else if ( (unsigned __int8)sub_CE8830(v51) )
        {
          if ( byte_5041108 != 1 && v82 <= 0x1F )
          {
            a2 = (__int64 *)v53;
            v54 = sub_36F9790(v52, v53, 3u, v74);
            if ( v54 )
            {
              ++v82;
              v85 = v54;
            }
          }
        }
        v51 += 40;
        ++v53;
      }
      while ( v50 != v51 );
      v2 = v52;
      v5 = v79;
    }
  }
  if ( !byte_5041108 )
  {
    v75 = v2 + 40;
    v78 = (__int64 *)v2[41];
    if ( v78 != v2 + 40 )
    {
      v9 = v5;
      while ( 1 )
      {
        v10 = v78[7];
        v83 = v78 + 6;
        if ( (__int64 *)v10 != v78 + 6 )
          break;
LABEL_13:
        v78 = (__int64 *)v78[1];
        if ( v75 == v78 )
          goto LABEL_14;
      }
      while ( 1 )
      {
        if ( *(_WORD *)(v10 + 68) != 7053 )
          goto LABEL_11;
        v14 = *(_QWORD *)(v10 + 32);
        v88 = 0;
        v15 = *(_QWORD *)(v14 + 64);
        v16 = *(unsigned int *)(v14 + 8);
        v89 = 0;
        v90 = 0;
        v80 = v15;
        v91 = 0;
        if ( (int)v16 < 0 )
          v17 = *(_QWORD *)(*(_QWORD *)(v76 + 56) + 16 * (v16 & 0x7FFFFFFF) + 8);
        else
          v17 = *(_QWORD *)(*(_QWORD *)(v76 + 304) + 8 * v16);
        if ( !v17 )
          goto LABEL_33;
        if ( (*(_BYTE *)(v17 + 3) & 0x10) != 0 )
          break;
LABEL_23:
        v86 = 1;
        v18 = 0;
        v19 = 0;
LABEL_24:
        v20 = *(_QWORD *)(v17 + 16);
        if ( (*(_BYTE *)(*(_QWORD *)(v20 + 16) + 32LL) & 0x40) == 0 )
        {
          v86 = 0;
          goto LABEL_27;
        }
        v30 = *(_QWORD *)(v20 + 32) + 200LL;
        if ( !(_DWORD)v18 )
        {
          ++v88;
          goto LABEL_70;
        }
        v31 = (v18 - 1) & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
        v32 = &v19[v31];
        v33 = *v32;
        if ( v30 == *v32 )
          goto LABEL_27;
        v34 = 1;
        v35 = 0;
        while ( 1 )
        {
          if ( v33 == -4096 )
          {
            if ( !v35 )
              v35 = v32;
            ++v88;
            v36 = v90 + 1;
            if ( 4 * ((int)v90 + 1) >= (unsigned int)(3 * v18) )
            {
LABEL_70:
              sub_36F88C0((__int64)&v88, 2 * v18);
              if ( (_DWORD)v91 )
              {
                v41 = (v91 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v35 = &v89[v41];
                v36 = v90 + 1;
                v42 = *v35;
                if ( v30 != *v35 )
                {
                  v43 = 1;
                  v44 = 0;
                  while ( v42 != -4096 )
                  {
                    if ( !v44 && v42 == -8192 )
                      v44 = v35;
                    v41 = (v91 - 1) & (v43 + v41);
                    v35 = &v89[v41];
                    v42 = *v35;
                    if ( v30 == *v35 )
                      goto LABEL_52;
                    ++v43;
                  }
                  if ( v44 )
                    v35 = v44;
                }
LABEL_52:
                LODWORD(v90) = v36;
                if ( *v35 != -4096 )
                  --HIDWORD(v90);
                *v35 = v30;
                v19 = v89;
                v18 = (unsigned int)v91;
                break;
              }
            }
            else
            {
              if ( (int)v18 - (v36 + HIDWORD(v90)) > (unsigned int)v18 >> 3 )
                goto LABEL_52;
              sub_36F88C0((__int64)&v88, v18);
              if ( (_DWORD)v91 )
              {
                v45 = 0;
                v46 = (v91 - 1) & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
                v35 = &v89[v46];
                v36 = v90 + 1;
                v47 = 1;
                v48 = *v35;
                if ( v30 != *v35 )
                {
                  while ( v48 != -4096 )
                  {
                    if ( v48 == -8192 && !v45 )
                      v45 = v35;
                    v46 = (v91 - 1) & (v47 + v46);
                    v35 = &v89[v46];
                    v48 = *v35;
                    if ( v30 == *v35 )
                      goto LABEL_52;
                    ++v47;
                  }
                  if ( v45 )
                    v35 = v45;
                }
                goto LABEL_52;
              }
            }
            LODWORD(v90) = v90 + 1;
            BUG();
          }
          if ( v33 != -8192 || v35 )
            v32 = v35;
          v31 = (v18 - 1) & (v34 + v31);
          v33 = v19[v31];
          if ( v30 == v33 )
            break;
          ++v34;
          v35 = v32;
          v32 = &v19[v31];
        }
LABEL_27:
        while ( 1 )
        {
          v17 = *(_QWORD *)(v17 + 32);
          if ( !v17 )
            break;
          if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
            goto LABEL_24;
        }
        v21 = &v19[v18];
        if ( (_DWORD)v90 && v19 != v21 )
        {
          while ( *v19 == -8192 || *v19 == -4096 )
          {
            if ( ++v19 == v21 )
              goto LABEL_29;
          }
          if ( v19 != v21 )
          {
            v37 = v80;
            v81 = v10;
LABEL_63:
            v38 = *v19;
            sub_2EAB460(*v19, v37, 0, 0);
            v39 = sub_36F75C0(*(unsigned __int16 *)(*(_QWORD *)(v38 + 16) + 68LL));
            sub_2E88D70(v40, (unsigned __int16 *)(*(_QWORD *)(v9 + 8) - 40LL * v39));
            while ( ++v19 != v21 )
            {
              if ( *v19 != -8192 && *v19 != -4096 )
              {
                if ( v21 != v19 )
                  goto LABEL_63;
                break;
              }
            }
            v10 = v81;
          }
        }
LABEL_29:
        if ( v86 )
          goto LABEL_33;
LABEL_30:
        sub_C7D6A0((__int64)v89, 8LL * (unsigned int)v91, 8);
LABEL_11:
        if ( (*(_BYTE *)v10 & 4) != 0 )
        {
          v10 = *(_QWORD *)(v10 + 8);
          if ( v83 == (__int64 *)v10 )
            goto LABEL_13;
        }
        else
        {
          while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
            v10 = *(_QWORD *)(v10 + 8);
          v10 = *(_QWORD *)(v10 + 8);
          if ( v83 == (__int64 *)v10 )
            goto LABEL_13;
        }
      }
      while ( 1 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( !v17 )
          break;
        if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
          goto LABEL_23;
      }
LABEL_33:
      v22 = *(_DWORD *)(a1 + 224);
      if ( v22 )
      {
        v23 = *(_QWORD *)(a1 + 208);
        v24 = (v22 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v25 = (__int64 *)(v23 + 8LL * v24);
        v26 = *v25;
        if ( *v25 == v10 )
          goto LABEL_30;
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( !v28 && v26 == -8192 )
            v28 = v25;
          v24 = (v22 - 1) & (v27 + v24);
          v25 = (__int64 *)(v23 + 8LL * v24);
          v26 = *v25;
          if ( v10 == *v25 )
            goto LABEL_30;
          ++v27;
        }
        if ( v28 )
          v25 = v28;
        ++*(_QWORD *)(a1 + 200);
        v29 = *(_DWORD *)(a1 + 216) + 1;
        if ( 4 * v29 < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a1 + 220) - v29 > v22 >> 3 )
          {
LABEL_41:
            *(_DWORD *)(a1 + 216) = v29;
            if ( *v25 != -4096 )
              --*(_DWORD *)(a1 + 220);
            *v25 = v10;
            goto LABEL_30;
          }
          sub_2E36C70(v74, v22);
          v67 = *(_DWORD *)(a1 + 224);
          if ( v67 )
          {
            v68 = v67 - 1;
            v69 = *(_QWORD *)(a1 + 208);
            v70 = 0;
            v71 = v68 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v29 = *(_DWORD *)(a1 + 216) + 1;
            v72 = 1;
            v25 = (__int64 *)(v69 + 8LL * v71);
            v73 = *v25;
            if ( *v25 != v10 )
            {
              while ( v73 != -4096 )
              {
                if ( v73 == -8192 && !v70 )
                  v70 = v25;
                v71 = v68 & (v72 + v71);
                v25 = (__int64 *)(v69 + 8LL * v71);
                v73 = *v25;
                if ( v10 == *v25 )
                  goto LABEL_41;
                ++v72;
              }
              if ( v70 )
                v25 = v70;
            }
            goto LABEL_41;
          }
LABEL_165:
          ++*(_DWORD *)(a1 + 216);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 200);
      }
      sub_2E36C70(v74, 2 * v22);
      v60 = *(_DWORD *)(a1 + 224);
      if ( v60 )
      {
        v61 = v60 - 1;
        v62 = *(_QWORD *)(a1 + 208);
        v63 = v61 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v29 = *(_DWORD *)(a1 + 216) + 1;
        v25 = (__int64 *)(v62 + 8LL * v63);
        v64 = *v25;
        if ( *v25 != v10 )
        {
          v65 = 1;
          v66 = 0;
          while ( v64 != -4096 )
          {
            if ( !v66 && v64 == -8192 )
              v66 = v25;
            v63 = v61 & (v65 + v63);
            v25 = (__int64 *)(v62 + 8LL * v63);
            v64 = *v25;
            if ( v10 == *v25 )
              goto LABEL_41;
            ++v65;
          }
          if ( v66 )
            v25 = v66;
        }
        goto LABEL_41;
      }
      goto LABEL_165;
    }
  }
LABEL_14:
  v11 = *(__int64 **)(a1 + 208);
  v12 = &v11[*(unsigned int *)(a1 + 224)];
  if ( *(_DWORD *)(a1 + 216) && v11 != v12 )
  {
    while ( *v11 == -4096 || *v11 == -8192 )
    {
      if ( v12 == ++v11 )
        return v85;
    }
    while ( v12 != v11 )
    {
      v56 = *v11++;
      sub_2E88E20(v56);
      if ( v11 == v12 )
        break;
      while ( *v11 == -8192 || *v11 == -4096 )
      {
        if ( v12 == ++v11 )
          return v85;
      }
    }
  }
  return v85;
}
