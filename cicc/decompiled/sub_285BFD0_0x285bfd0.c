// Function: sub_285BFD0
// Address: 0x285bfd0
//
void __fastcall sub_285BFD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r10d
  __int64 *v8; // rdx
  __int64 v12; // r15
  __int64 v13; // rsi
  int v14; // ecx
  int v15; // ecx
  unsigned int v16; // eax
  __int64 *v17; // rdi
  __int64 **v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r15
  __int64 v22; // r9
  __int64 **v23; // r14
  int v24; // ecx
  __int64 *v25; // r11
  __int64 v26; // rax
  int v27; // ecx
  unsigned int v28; // edx
  __int64 *v29; // rsi
  int v30; // edi
  unsigned __int64 v31; // r13
  int v32; // eax
  int v33; // edx
  __int64 v34; // r13
  unsigned __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // r15
  __int64 v38; // r10
  __int64 v39; // r14
  __int64 v40; // rax
  char v41; // dl
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // rcx
  char v44; // al
  int v45; // r8d
  int v46; // eax
  __int64 *v47; // r13
  __int64 v48; // rdi
  unsigned int v49; // edx
  unsigned int v50; // eax
  unsigned int v51; // eax
  unsigned int v52; // ecx
  unsigned int v53; // eax
  int v54; // eax
  __int64 *v55; // r14
  char v56; // al
  unsigned __int8 v57; // al
  unsigned int v58; // eax
  __int64 v59; // r15
  __int64 v60; // rcx
  __int64 v61; // r15
  __int64 v62; // r11
  __int64 v63; // r8
  int v64; // edx
  __int64 v65; // rax
  int v66; // edx
  __int64 v67; // [rsp+0h] [rbp-70h]
  unsigned int v68; // [rsp+8h] [rbp-68h]
  __int64 v69; // [rsp+8h] [rbp-68h]
  char v70; // [rsp+8h] [rbp-68h]
  __int64 v71; // [rsp+8h] [rbp-68h]
  __int64 **v72; // [rsp+10h] [rbp-60h]
  unsigned int v73; // [rsp+10h] [rbp-60h]
  unsigned int v74; // [rsp+18h] [rbp-58h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  unsigned int v76; // [rsp+18h] [rbp-58h]
  unsigned int v77; // [rsp+18h] [rbp-58h]
  unsigned int v78; // [rsp+18h] [rbp-58h]
  unsigned int v79; // [rsp+18h] [rbp-58h]
  unsigned int v80; // [rsp+18h] [rbp-58h]
  int v81; // [rsp+18h] [rbp-58h]
  unsigned int v82; // [rsp+18h] [rbp-58h]
  int v83; // [rsp+20h] [rbp-50h]
  int v84; // [rsp+24h] [rbp-4Ch]
  unsigned __int64 v86; // [rsp+30h] [rbp-40h] BYREF
  int v87; // [rsp+38h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 28);
  if ( v6 != -1 )
  {
    v8 = *(__int64 **)(a2 + 88);
    v12 = a6;
    v84 = *(_DWORD *)(a1 + 32);
    v83 = *(_DWORD *)(a1 + 40);
    if ( !v8 )
      goto LABEL_8;
    v13 = *(_QWORD *)(a4 + 8);
    v14 = *(_DWORD *)(a4 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = v15 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v17 = *(__int64 **)(v13 + 8LL * v16);
      if ( v8 == v17 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 24) = -1;
        *(_QWORD *)(a1 + 32) = -1;
        *(_QWORD *)(a1 + 40) = -1;
        *(_QWORD *)(a1 + 48) = -1;
        return;
      }
      v45 = 1;
      while ( v17 != (__int64 *)-4096LL )
      {
        a6 = (unsigned int)(v45 + 1);
        v16 = v15 & (v45 + v16);
        v17 = *(__int64 **)(v13 + 8LL * v16);
        if ( v8 == v17 )
          goto LABEL_5;
        ++v45;
      }
    }
    v74 = v6;
    sub_285BE30(a1, a2, v8, a3, v12, a6);
    v6 = v74;
    if ( *(_DWORD *)(a1 + 28) != -1 )
    {
LABEL_8:
      v18 = *(__int64 ***)(a2 + 40);
      v19 = *(unsigned int *)(a2 + 48);
      v20 = v12;
      v21 = a3;
      v22 = (__int64)&v18[v19];
      v23 = v18;
      if ( v18 != (__int64 **)v22 )
      {
        do
        {
          v24 = *(_DWORD *)(a4 + 24);
          v25 = *v23;
          v26 = *(_QWORD *)(a4 + 8);
          if ( v24 )
          {
            v27 = v24 - 1;
            v28 = v27 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v29 = *(__int64 **)(v26 + 8LL * v28);
            if ( v25 == v29 )
              goto LABEL_5;
            v30 = 1;
            while ( v29 != (__int64 *)-4096LL )
            {
              v28 = v27 & (v30 + v28);
              v29 = *(__int64 **)(v26 + 8LL * v28);
              if ( v25 == v29 )
                goto LABEL_5;
              ++v30;
            }
          }
          v68 = v6;
          v72 = (__int64 **)v22;
          v75 = v20;
          sub_285BE30(a1, a2, v25, v21, v20, v22);
          if ( *(_DWORD *)(a1 + 28) == -1 )
            return;
          v22 = (__int64)v72;
          ++v23;
          v20 = v75;
          v6 = v68;
        }
        while ( v72 != v23 );
        v19 = *(unsigned int *)(a2 + 48);
      }
      v31 = v19 - ((*(_QWORD *)(a2 + 88) == 0) - 1LL);
      if ( v31 <= 1 )
      {
        v33 = *(_DWORD *)(a1 + 40);
      }
      else
      {
        v32 = 1;
        if ( *(_QWORD *)(a2 + 32) )
        {
          v80 = v6;
          v57 = sub_2853690(*(__int64 **)(a1 + 16), a5, a2);
          v6 = v80;
          v32 = v57 + 1;
        }
        v33 = *(_DWORD *)(a1 + 40) + v31 - v32;
        *(_DWORD *)(a1 + 40) = v33;
      }
      LODWORD(v34) = 0;
      *(_DWORD *)(a1 + 40) = v33 - ((*(_QWORD *)(a2 + 96) == 0) - 1);
      v35 = *(_QWORD *)(a2 + 32);
      if ( v35 )
      {
        v55 = *(__int64 **)(a1 + 16);
        v79 = v6;
        v56 = sub_2850670(
                v55,
                *(_QWORD *)(a5 + 712),
                *(_BYTE *)(a5 + 720),
                *(_QWORD *)(a5 + 728),
                *(_BYTE *)(a5 + 736),
                *(_DWORD *)(a5 + 32),
                *(_QWORD *)(a5 + 40),
                *(unsigned int *)(a5 + 48),
                *(_QWORD *)a2,
                *(_QWORD *)(a2 + 8),
                *(_BYTE *)(a2 + 16),
                *(_BYTE *)(a2 + 24),
                v35);
        v6 = v79;
        if ( v56 )
        {
          v58 = *(_DWORD *)(a5 + 32);
          if ( v58 == 2 )
          {
            v59 = *(_QWORD *)(a2 + 8);
            v60 = v59 + *(_QWORD *)(a5 + 712);
            v61 = *(_QWORD *)(a5 + 728) + v59;
            if ( *(_BYTE *)(a2 + 16) )
            {
              v62 = v61;
              v63 = v60;
              v61 = 0;
              v60 = 0;
            }
            else
            {
              v62 = 0;
              v63 = 0;
            }
            v73 = v79;
            v71 = v62;
            v34 = sub_DFA7B0(
                    v55,
                    *(_QWORD *)(a5 + 40),
                    *(_QWORD *)a2,
                    v60,
                    v63,
                    *(_BYTE *)(a2 + 24),
                    *(_QWORD *)(a2 + 32));
            v81 = v64;
            v65 = sub_DFA7B0(
                    v55,
                    *(_QWORD *)(a5 + 40),
                    *(_QWORD *)a2,
                    v61,
                    v71,
                    *(_BYTE *)(a2 + 24),
                    *(_QWORD *)(a2 + 32));
            v6 = v73;
            if ( v81 == v66 )
            {
              if ( v34 < v65 )
                LODWORD(v34) = v65;
            }
            else if ( v81 < v66 )
            {
              LODWORD(v34) = v65;
            }
            else
            {
              v66 = v81;
            }
            if ( v66 )
              LODWORD(v34) = 0;
          }
          else if ( v58 >= 4 )
          {
            BUG();
          }
        }
        else
        {
          LODWORD(v34) = *(_QWORD *)(a2 + 32) != 1;
        }
      }
      *(_DWORD *)(a1 + 52) += v34;
      v36 = *(_QWORD *)(a5 + 56);
      if ( v36 + 80LL * *(unsigned int *)(a5 + 64) != v36 )
      {
        v76 = v6;
        v37 = v36 + 80LL * *(unsigned int *)(a5 + 64);
        v38 = a5;
        while ( 1 )
        {
          v39 = *(_QWORD *)(v36 + 64);
          v40 = *(_QWORD *)(a2 + 8);
          v41 = *(_BYTE *)(v36 + 72);
          if ( v39 && v40 && *(_BYTE *)(a2 + 16) != v41 )
          {
            *(_DWORD *)(a1 + 44) += 2048;
          }
          else
          {
            v42 = v40 + v39;
            if ( !v41 )
              v41 = *(_BYTE *)(a2 + 16);
            if ( *(_QWORD *)a2 )
            {
              *(_DWORD *)(a1 + 44) += 64;
              if ( *(_DWORD *)(v38 + 32) == 2 && v42 )
                goto LABEL_35;
            }
            else if ( v42 )
            {
              v67 = v38;
              v70 = v41;
              v87 = 64;
              v86 = v42;
              v46 = sub_969260((__int64)&v86);
              v38 = v67;
              v41 = v70;
              *(_DWORD *)(a1 + 44) = *(_DWORD *)(a1 + 44) + 65 - v46;
              if ( *(_DWORD *)(v67 + 32) == 2 )
              {
LABEL_35:
                v43 = 0;
                if ( !v41 )
                  v43 = v42;
                v69 = v38;
                v44 = sub_DFA150(
                        *(__int64 **)(a1 + 16),
                        *(_QWORD *)(v38 + 40),
                        *(_QWORD *)a2,
                        v43,
                        *(_BYTE *)(a2 + 24),
                        *(_QWORD *)(a2 + 32));
                v38 = v69;
                if ( !v44 )
                  ++*(_DWORD *)(a1 + 40);
              }
            }
          }
          v36 += 80;
          if ( v37 == v36 )
          {
            v6 = v76;
            break;
          }
        }
      }
      if ( byte_5001808 )
      {
        v47 = *(__int64 **)(a1 + 16);
        if ( *(_DWORD *)(a2 + 48) )
        {
          v82 = v6;
          sub_D95540(**(_QWORD **)(a2 + 40));
          v6 = v82;
        }
        else
        {
          v48 = *(_QWORD *)(a2 + 88);
          if ( v48 )
          {
            v77 = v6;
            sub_D95540(v48);
            v6 = v77;
          }
        }
        v78 = v6;
        sub_DFB180(v47, 0);
        v49 = sub_DFB120((__int64)v47) - 1;
        v50 = *(_DWORD *)(a1 + 28);
        if ( v50 > v49 )
        {
          v51 = *(_DWORD *)(a1 + 24) + v50;
          v52 = v51 - v49;
          v53 = v51 - v78;
          if ( v49 >= v78 )
            v53 = v52;
          *(_DWORD *)(a1 + 24) = v53;
        }
        if ( *(_DWORD *)(a5 + 32) == 3
          && (*(_QWORD *)(a2 + 96) || *(_QWORD *)(a2 + 8) || *(_DWORD *)(a2 + 48) != 1 || *(_QWORD *)(a2 + 88))
          && !(unsigned __int8)sub_DFA2D0(*(_QWORD *)(a1 + 16)) )
        {
          ++*(_DWORD *)(a1 + 24);
        }
        v54 = *(_DWORD *)(a1 + 24) + *(_DWORD *)(a1 + 32) - v84;
        *(_DWORD *)(a1 + 24) = v54;
        if ( *(_DWORD *)(a5 + 32) != 3 )
          *(_DWORD *)(a1 + 24) = *(_DWORD *)(a1 + 40) + v54 - v83;
      }
    }
  }
}
