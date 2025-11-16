// Function: sub_3016110
// Address: 0x3016110
//
void __fastcall sub_3016110(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // zf
  int v10; // eax
  unsigned __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // r12
  unsigned int v14; // esi
  __int64 v15; // rcx
  int v16; // r9d
  __int64 *v17; // r8
  unsigned int v18; // edx
  __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // r14
  int v24; // ecx
  __int64 v25; // r14
  unsigned __int64 v26; // rcx
  __int64 v27; // rdi
  _QWORD *v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // r14
  int v31; // eax
  unsigned int v32; // r13d
  int v33; // r15d
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rbx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  int v41; // edx
  __int64 v42; // rax
  unsigned int v43; // eax
  unsigned int v44; // esi
  __int64 v45; // r9
  int v46; // r12d
  __int64 *v47; // r8
  __int64 v48; // rcx
  __int64 *v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rax
  int v52; // eax
  int v53; // ecx
  __int64 v54; // rsi
  unsigned int v55; // eax
  __int64 v56; // rdi
  int v57; // r9d
  __int64 *v58; // r8
  int v59; // eax
  int v60; // eax
  __int64 v61; // rsi
  int v62; // r8d
  unsigned int v63; // r14d
  __int64 *v64; // rdi
  __int64 v65; // rcx
  _QWORD *v66; // rax
  int v67; // ecx
  int v68; // r11d
  int v69; // r11d
  __int64 v70; // r9
  int v71; // edi
  unsigned int v72; // ebx
  __int64 *v73; // rsi
  __int64 v74; // r8
  int v75; // ebx
  int v76; // ebx
  __int64 v77; // r10
  unsigned int v78; // esi
  __int64 v79; // r9
  int v80; // r8d
  __int64 *v81; // rdi
  __int64 v82; // [rsp+8h] [rbp-C8h]
  __int64 v83; // [rsp+8h] [rbp-C8h]
  __int64 v84; // [rsp+10h] [rbp-C0h]
  __int64 v85; // [rsp+18h] [rbp-B8h]
  int v87; // [rsp+38h] [rbp-98h]
  __int64 v88; // [rsp+38h] [rbp-98h]
  __int64 v89; // [rsp+48h] [rbp-88h] BYREF
  _BYTE *v90; // [rsp+50h] [rbp-80h] BYREF
  __int64 v91; // [rsp+58h] [rbp-78h]
  _BYTE v92[112]; // [rsp+60h] [rbp-70h] BYREF

  v90 = v92;
  v91 = 0x800000000LL;
  v3 = sub_22077B0(0x10u);
  v6 = v3;
  if ( v3 )
  {
    *(_QWORD *)v3 = a1;
    *(_DWORD *)(v3 + 8) = a2;
  }
  v7 = (unsigned int)v91;
  v8 = (unsigned int)v91 + 1LL;
  if ( v8 > HIDWORD(v91) )
  {
    sub_C8D5F0((__int64)&v90, v92, v8, 8u, v4, v5);
    v7 = (unsigned int)v91;
  }
  *(_QWORD *)&v90[8 * v7] = v6;
  v9 = (_DWORD)v91 == -1;
  v10 = v91 + 1;
  LODWORD(v91) = v91 + 1;
  if ( !v9 )
  {
    v85 = a3 + 128;
    v84 = a3 + 64;
    while ( 1 )
    {
      v11 = *(_QWORD *)&v90[8 * v10 - 8];
      LODWORD(v91) = v10 - 1;
      v12 = *(_QWORD *)v11;
      v13 = *(int *)(v11 + 8);
      j_j___libc_free_0(v11);
      v14 = *(_DWORD *)(a3 + 152);
      if ( !v14 )
        break;
      v15 = *(_QWORD *)(a3 + 136);
      v16 = 1;
      v17 = 0;
      v18 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v19 = (__int64 *)(v15 + 16LL * v18);
      v20 = *v19;
      if ( v12 == *v19 )
      {
LABEL_10:
        v10 = v91;
        if ( *((_DWORD *)v19 + 2) > (int)v13 )
          goto LABEL_11;
LABEL_7:
        if ( !v10 )
          goto LABEL_57;
      }
      else
      {
        while ( v20 != -4096 )
        {
          if ( v20 == -8192 && !v17 )
            v17 = v19;
          v18 = (v14 - 1) & (v16 + v18);
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v12 == *v19 )
            goto LABEL_10;
          ++v16;
        }
        if ( v17 )
          v19 = v17;
        ++*(_QWORD *)(a3 + 128);
        v41 = *(_DWORD *)(a3 + 144) + 1;
        if ( 4 * v41 < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a3 + 148) - v41 <= v14 >> 3 )
          {
            sub_FF1B10(v85, v14);
            v59 = *(_DWORD *)(a3 + 152);
            if ( !v59 )
            {
LABEL_125:
              ++*(_DWORD *)(a3 + 144);
              BUG();
            }
            v60 = v59 - 1;
            v61 = *(_QWORD *)(a3 + 136);
            v62 = 1;
            v63 = v60 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v41 = *(_DWORD *)(a3 + 144) + 1;
            v64 = 0;
            v19 = (__int64 *)(v61 + 16LL * v63);
            v65 = *v19;
            if ( v12 != *v19 )
            {
              while ( v65 != -4096 )
              {
                if ( v65 == -8192 && !v64 )
                  v64 = v19;
                v63 = v60 & (v62 + v63);
                v19 = (__int64 *)(v61 + 16LL * v63);
                v65 = *v19;
                if ( v12 == *v19 )
                  goto LABEL_42;
                ++v62;
              }
              if ( v64 )
                v19 = v64;
            }
          }
          goto LABEL_42;
        }
LABEL_63:
        sub_FF1B10(v85, 2 * v14);
        v52 = *(_DWORD *)(a3 + 152);
        if ( !v52 )
          goto LABEL_125;
        v53 = v52 - 1;
        v54 = *(_QWORD *)(a3 + 136);
        v55 = (v52 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v41 = *(_DWORD *)(a3 + 144) + 1;
        v19 = (__int64 *)(v54 + 16LL * v55);
        v56 = *v19;
        if ( v12 != *v19 )
        {
          v57 = 1;
          v58 = 0;
          while ( v56 != -4096 )
          {
            if ( !v58 && v56 == -8192 )
              v58 = v19;
            v55 = v53 & (v57 + v55);
            v19 = (__int64 *)(v54 + 16LL * v55);
            v56 = *v19;
            if ( v12 == *v19 )
              goto LABEL_42;
            ++v57;
          }
          if ( v58 )
            v19 = v58;
        }
LABEL_42:
        *(_DWORD *)(a3 + 144) = v41;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a3 + 148);
        *v19 = v12;
        *((_DWORD *)v19 + 2) = 0;
LABEL_11:
        v21 = sub_AA4FF0(v12);
        v22 = v12 + 48;
        v23 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v12 + 48 == v23 )
        {
          v25 = 0;
        }
        else
        {
          if ( !v23 )
            goto LABEL_126;
          v24 = *(unsigned __int8 *)(v23 - 24);
          v25 = v23 - 24;
          if ( (unsigned int)(v24 - 30) >= 0xB )
            v25 = 0;
        }
        if ( !v21 )
LABEL_126:
          BUG();
        v26 = (unsigned int)*(unsigned __int8 *)(v21 - 24) - 39;
        if ( (unsigned int)v26 <= 0x38 )
        {
          v27 = 0x100060000000001LL;
          if ( _bittest64(&v27, v26) )
          {
            v89 = v21 - 24;
            v28 = sub_3014430(a3, &v89);
            v22 = v12 + 48;
            v13 = *(int *)v28;
          }
        }
        *((_DWORD *)v19 + 2) = v13;
        if ( (unsigned __int8)(*(_BYTE *)v25 - 37) <= 1u )
        {
          if ( (int)v13 > 0 )
            LODWORD(v13) = *(_DWORD *)(*(_QWORD *)(a3 + 160) + 16 * v13);
LABEL_22:
          v29 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v22 == v29 )
            goto LABEL_56;
          goto LABEL_23;
        }
        if ( *(_BYTE *)v25 != 34 )
          goto LABEL_22;
        v42 = *(_QWORD *)(v25 - 32);
        if ( !v42
          || *(_BYTE *)v42
          || *(_QWORD *)(v42 + 24) != *(_QWORD *)(v25 + 80)
          || (*(_BYTE *)(v42 + 33) & 0x20) == 0 )
        {
          goto LABEL_22;
        }
        v43 = *(_DWORD *)(v42 + 36) & 0xFFFFFFFD;
        if ( v43 == 316 )
        {
          v88 = v22;
          v89 = v25;
          v66 = sub_3013480(v84, &v89);
          v22 = v88;
          LODWORD(v13) = *(_DWORD *)v66;
          goto LABEL_22;
        }
        if ( v43 != 317 )
          goto LABEL_22;
        v44 = *(_DWORD *)(a3 + 88);
        if ( !v44 )
        {
          ++*(_QWORD *)(a3 + 64);
LABEL_97:
          v83 = v22;
          sub_30132A0(v84, 2 * v44);
          v75 = *(_DWORD *)(a3 + 88);
          if ( !v75 )
          {
LABEL_124:
            ++*(_DWORD *)(a3 + 80);
            BUG();
          }
          v76 = v75 - 1;
          v77 = *(_QWORD *)(a3 + 72);
          v22 = v83;
          v78 = v76 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v67 = *(_DWORD *)(a3 + 80) + 1;
          v49 = (__int64 *)(v77 + 16LL * v78);
          v79 = *v49;
          if ( *v49 != v25 )
          {
            v80 = 1;
            v81 = 0;
            while ( v79 != -4096 )
            {
              if ( v79 == -8192 && !v81 )
                v81 = v49;
              v78 = v76 & (v80 + v78);
              v49 = (__int64 *)(v77 + 16LL * v78);
              v79 = *v49;
              if ( v25 == *v49 )
                goto LABEL_87;
              ++v80;
            }
            if ( v81 )
              v49 = v81;
          }
          goto LABEL_87;
        }
        v45 = *(_QWORD *)(a3 + 72);
        v46 = 1;
        v47 = 0;
        LODWORD(v48) = (v44 - 1) & (((unsigned int)v25 >> 4) ^ ((unsigned int)v25 >> 9));
        v49 = (__int64 *)(v45 + 16LL * (unsigned int)v48);
        v50 = *v49;
        if ( v25 == *v49 )
        {
LABEL_54:
          v51 = 16LL * *((int *)v49 + 2);
          goto LABEL_55;
        }
        while ( v50 != -4096 )
        {
          if ( v50 == -8192 && !v47 )
            v47 = v49;
          v48 = (v44 - 1) & ((_DWORD)v48 + v46);
          v49 = (__int64 *)(v45 + 16 * v48);
          v50 = *v49;
          if ( v25 == *v49 )
            goto LABEL_54;
          ++v46;
        }
        if ( v47 )
          v49 = v47;
        ++*(_QWORD *)(a3 + 64);
        v67 = *(_DWORD *)(a3 + 80) + 1;
        if ( 4 * v67 >= 3 * v44 )
          goto LABEL_97;
        if ( v44 - *(_DWORD *)(a3 + 84) - v67 <= v44 >> 3 )
        {
          v82 = v22;
          sub_30132A0(v84, v44);
          v68 = *(_DWORD *)(a3 + 88);
          if ( !v68 )
            goto LABEL_124;
          v69 = v68 - 1;
          v70 = *(_QWORD *)(a3 + 72);
          v71 = 1;
          v72 = v69 & (((unsigned int)v25 >> 4) ^ ((unsigned int)v25 >> 9));
          v22 = v82;
          v67 = *(_DWORD *)(a3 + 80) + 1;
          v73 = 0;
          v49 = (__int64 *)(v70 + 16LL * v72);
          v74 = *v49;
          if ( v25 != *v49 )
          {
            while ( v74 != -4096 )
            {
              if ( !v73 && v74 == -8192 )
                v73 = v49;
              v72 = v69 & (v71 + v72);
              v49 = (__int64 *)(v70 + 16LL * v72);
              v74 = *v49;
              if ( v25 == *v49 )
                goto LABEL_87;
              ++v71;
            }
            if ( v73 )
              v49 = v73;
          }
        }
LABEL_87:
        *(_DWORD *)(a3 + 80) = v67;
        if ( *v49 != -4096 )
          --*(_DWORD *)(a3 + 84);
        *v49 = v25;
        *((_DWORD *)v49 + 2) = 0;
        v51 = 0;
LABEL_55:
        LODWORD(v13) = *(_DWORD *)(*(_QWORD *)(a3 + 160) + v51);
        v29 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v22 == v29 )
          goto LABEL_56;
LABEL_23:
        if ( !v29 )
          BUG();
        v30 = v29 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 <= 0xA )
        {
          v31 = sub_B46E30(v30);
          if ( v31 )
          {
            v87 = v31;
            v32 = 0;
            v33 = v13;
            do
            {
              v34 = sub_B46EC0(v30, v32);
              v35 = sub_22077B0(0x10u);
              v38 = v35;
              if ( v35 )
              {
                *(_QWORD *)v35 = v34;
                *(_DWORD *)(v35 + 8) = v33;
              }
              v39 = (unsigned int)v91;
              v40 = (unsigned int)v91 + 1LL;
              if ( v40 > HIDWORD(v91) )
              {
                sub_C8D5F0((__int64)&v90, v92, v40, 8u, v36, v37);
                v39 = (unsigned int)v91;
              }
              ++v32;
              *(_QWORD *)&v90[8 * v39] = v38;
              v10 = v91 + 1;
              LODWORD(v91) = v91 + 1;
            }
            while ( v87 != v32 );
            goto LABEL_7;
          }
        }
LABEL_56:
        v10 = v91;
        if ( !(_DWORD)v91 )
          goto LABEL_57;
      }
    }
    ++*(_QWORD *)(a3 + 128);
    goto LABEL_63;
  }
LABEL_57:
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
}
