// Function: sub_CE5730
// Address: 0xce5730
//
__int64 __fastcall sub_CE5730(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r11
  unsigned int v15; // esi
  __int64 v16; // r15
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 *v22; // r13
  __int64 *v23; // r15
  unsigned int v24; // r8d
  __int64 *v25; // rcx
  __int64 v26; // r10
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // rdi
  unsigned int v30; // esi
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // r9
  __int64 *v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // rdi
  unsigned int v38; // esi
  __int64 v39; // r10
  __int64 v40; // rdx
  __int64 v41; // r9
  __int64 *v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rdi
  int v46; // ecx
  int v47; // edx
  int v48; // esi
  int v49; // esi
  __int64 v50; // r10
  int v51; // r11d
  int v52; // esi
  int v53; // esi
  unsigned int v54; // eax
  __int64 *v55; // r10
  int v56; // esi
  int v57; // esi
  __int64 v58; // r10
  int v59; // r11d
  int v60; // esi
  int v61; // esi
  unsigned int v62; // eax
  __int64 v63; // r10
  int v64; // eax
  int v65; // eax
  int v66; // ecx
  __int64 v67; // rdi
  unsigned int v68; // edx
  __int64 v69; // rsi
  int v70; // ecx
  __int64 v71; // rdi
  unsigned int v72; // edx
  __int64 v73; // rsi
  __int64 v74; // [rsp+0h] [rbp-C0h]
  int v75; // [rsp+8h] [rbp-B8h]
  __int64 v76; // [rsp+8h] [rbp-B8h]
  unsigned int v77; // [rsp+8h] [rbp-B8h]
  __int64 v78; // [rsp+10h] [rbp-B0h]
  unsigned int v79; // [rsp+20h] [rbp-A0h]
  __int64 v80; // [rsp+28h] [rbp-98h]
  __int64 v81; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v82[2]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE v83[48]; // [rsp+50h] [rbp-70h] BYREF
  int v84; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 16);
  v5 = **(_QWORD **)(a2 + 32);
  v81 = v5;
  if ( v5 )
  {
    v6 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
    result = v6;
  }
  else
  {
    v6 = 0;
    result = 0;
  }
  if ( (unsigned int)result < *(_DWORD *)(v4 + 32) )
  {
    result = *(_QWORD *)(v4 + 24);
    if ( *(_QWORD *)(result + 8 * v6) )
    {
      v78 = a1 + 112;
      v12 = *sub_CE3FC0(a1 + 112, &v81);
      v82[0] = v83;
      v82[1] = 0x600000000LL;
      if ( *(_DWORD *)(v12 + 32) )
        sub_CE14D0((__int64)v82, v12 + 24, v8, v9, v10, v11);
      v84 = *(_DWORD *)(v12 + 88);
      v13 = *(_QWORD *)(v81 + 56);
      v14 = v81 + 48;
      v80 = a1 + 56;
      if ( v13 != v81 + 48 )
      {
        while ( 1 )
        {
          if ( !v13 )
            BUG();
          if ( *(_BYTE *)(v13 - 24) != 84 )
            goto LABEL_14;
          v15 = *(_DWORD *)(a1 + 80);
          v16 = v13 - 24;
          if ( !v15 )
            break;
          v11 = v15 - 1;
          v10 = *(_QWORD *)(a1 + 64);
          v17 = v11 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v18 = v10 + 16LL * v17;
          v19 = *(_QWORD *)v18;
          if ( v16 != *(_QWORD *)v18 )
          {
            v75 = 1;
            v63 = 0;
            while ( v19 != -4096 )
            {
              if ( !v63 && v19 == -8192 )
                v63 = v18;
              v17 = v11 & (v75 + v17);
              v10 = (unsigned int)(v75 + 1);
              v18 = *(_QWORD *)(a1 + 64) + 16LL * v17;
              v19 = *(_QWORD *)v18;
              if ( v16 == *(_QWORD *)v18 )
                goto LABEL_12;
              ++v75;
            }
            if ( !v63 )
              v63 = v18;
            v64 = *(_DWORD *)(a1 + 72);
            ++*(_QWORD *)(a1 + 56);
            v65 = v64 + 1;
            if ( 4 * v65 < 3 * v15 )
            {
              v9 = v15 - *(_DWORD *)(a1 + 76) - v65;
              if ( (unsigned int)v9 <= v15 >> 3 )
              {
                v74 = v14;
                v77 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
                sub_CE2410(v80, v15);
                v70 = *(_DWORD *)(a1 + 80);
                if ( !v70 )
                {
LABEL_143:
                  ++*(_DWORD *)(a1 + 72);
                  BUG();
                }
                v9 = (unsigned int)(v70 - 1);
                v71 = *(_QWORD *)(a1 + 64);
                v10 = 0;
                v14 = v74;
                v11 = 1;
                v72 = v9 & v77;
                v63 = v71 + 16LL * ((unsigned int)v9 & v77);
                v73 = *(_QWORD *)v63;
                v65 = *(_DWORD *)(a1 + 72) + 1;
                if ( v16 != *(_QWORD *)v63 )
                {
                  while ( v73 != -4096 )
                  {
                    if ( v73 == -8192 && !v10 )
                      v10 = v63;
                    v72 = v9 & (v11 + v72);
                    v63 = v71 + 16LL * v72;
                    v73 = *(_QWORD *)v63;
                    if ( v16 == *(_QWORD *)v63 )
                      goto LABEL_103;
                    v11 = (unsigned int)(v11 + 1);
                  }
                  goto LABEL_111;
                }
              }
              goto LABEL_103;
            }
LABEL_107:
            v76 = v14;
            sub_CE2410(v80, 2 * v15);
            v66 = *(_DWORD *)(a1 + 80);
            if ( !v66 )
              goto LABEL_143;
            v9 = (unsigned int)(v66 - 1);
            v67 = *(_QWORD *)(a1 + 64);
            v14 = v76;
            v68 = v9 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v63 = v67 + 16LL * v68;
            v69 = *(_QWORD *)v63;
            v65 = *(_DWORD *)(a1 + 72) + 1;
            if ( v16 != *(_QWORD *)v63 )
            {
              v11 = 1;
              v10 = 0;
              while ( v69 != -4096 )
              {
                if ( !v10 && v69 == -8192 )
                  v10 = v63;
                v68 = v9 & (v11 + v68);
                v63 = v67 + 16LL * v68;
                v69 = *(_QWORD *)v63;
                if ( v16 == *(_QWORD *)v63 )
                  goto LABEL_103;
                v11 = (unsigned int)(v11 + 1);
              }
LABEL_111:
              if ( v10 )
                v63 = v10;
            }
LABEL_103:
            *(_DWORD *)(a1 + 72) = v65;
            if ( *(_QWORD *)v63 != -4096 )
              --*(_DWORD *)(a1 + 76);
            *(_QWORD *)v63 = v16;
            v8 = -2;
            v20 = 0;
            *(_DWORD *)(v63 + 8) = 0;
            goto LABEL_13;
          }
LABEL_12:
          v9 = *(unsigned int *)(v18 + 8);
          v8 = ~(1LL << v9);
          v20 = 8LL * ((unsigned int)v9 >> 6);
LABEL_13:
          *(_QWORD *)(v82[0] + v20) &= v8;
          v13 = *(_QWORD *)(v13 + 8);
          if ( v14 == v13 )
            goto LABEL_14;
        }
        ++*(_QWORD *)(a1 + 56);
        goto LABEL_107;
      }
LABEL_14:
      v21 = (unsigned __int64)v82;
      result = sub_CE1BC0(v12 + 96, (__int64)v82, v8, v9, v10, v11);
      v22 = *(__int64 **)(a2 + 32);
      v23 = *(__int64 **)(a2 + 40);
      if ( v22 != v23 )
      {
        while ( 1 )
        {
          result = *(_QWORD *)(a1 + 8);
          v28 = *v22;
          v21 = *(unsigned int *)(result + 24);
          v29 = *(_QWORD *)(result + 8);
          if ( !(_DWORD)v21 )
            goto LABEL_22;
          v21 = (unsigned int)(v21 - 1);
          result = ((unsigned int)v28 >> 4) ^ ((unsigned int)v28 >> 9);
          v24 = v21 & (((unsigned int)v28 >> 4) ^ ((unsigned int)v28 >> 9));
          v25 = (__int64 *)(v29 + 16LL * v24);
          v26 = *v25;
          if ( v28 != *v25 )
          {
            v46 = 1;
            while ( v26 != -4096 )
            {
              v47 = v46 + 1;
              v24 = v21 & (v46 + v24);
              v25 = (__int64 *)(v29 + 16LL * v24);
              v26 = *v25;
              if ( v28 == *v25 )
                goto LABEL_17;
              v46 = v47;
            }
LABEL_22:
            if ( v28 != v81 )
              BUG();
            goto LABEL_20;
          }
LABEL_17:
          if ( v28 != v81 )
          {
            v27 = v25[1];
            if ( a2 == v27 || v28 == **(_QWORD **)(v27 + 32) )
              break;
          }
LABEL_20:
          if ( v23 == ++v22 )
            goto LABEL_24;
        }
        v30 = *(_DWORD *)(a1 + 136);
        if ( v30 )
        {
          v31 = *(_QWORD *)(a1 + 120);
          v32 = 1;
          v33 = (v30 - 1) & (unsigned int)result;
          v34 = 0;
          v35 = v31 + 16 * v33;
          v36 = *(_QWORD *)v35;
          if ( v28 == *(_QWORD *)v35 )
          {
LABEL_29:
            v37 = *(_QWORD *)(v35 + 8);
            goto LABEL_30;
          }
          while ( v36 != -4096 )
          {
            if ( v36 == -8192 && !v34 )
              v34 = (__int64 *)v35;
            v33 = (v30 - 1) & ((_DWORD)v32 + (_DWORD)v33);
            v35 = v31 + 16LL * (unsigned int)v33;
            v36 = *(_QWORD *)v35;
            if ( v28 == *(_QWORD *)v35 )
              goto LABEL_29;
            v32 = (unsigned int)(v32 + 1);
          }
          v32 = *(unsigned int *)(a1 + 128);
          if ( !v34 )
            v34 = (__int64 *)v35;
          ++*(_QWORD *)(a1 + 112);
          v35 = (unsigned int)(v32 + 1);
          if ( 4 * (int)v35 < 3 * v30 )
          {
            v36 = v30 - *(_DWORD *)(a1 + 132) - (unsigned int)v35;
            v33 = v30 >> 3;
            if ( (unsigned int)v36 > (unsigned int)v33 )
              goto LABEL_62;
            sub_CE3D80(v78, v30);
            v56 = *(_DWORD *)(a1 + 136);
            if ( !v56 )
              goto LABEL_144;
            LODWORD(result) = ((unsigned int)v28 >> 4) ^ ((unsigned int)v28 >> 9);
            v57 = v56 - 1;
            v58 = *(_QWORD *)(a1 + 120);
            v59 = 1;
            v36 = v57 & (unsigned int)result;
            v35 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
            v32 = 0;
            v34 = (__int64 *)(v58 + 16 * v36);
            v33 = *v34;
            if ( v28 == *v34 )
              goto LABEL_62;
            while ( v33 != -4096 )
            {
              if ( v33 == -8192 && !v32 )
                v32 = (__int64)v34;
              v36 = v57 & (unsigned int)(v59 + v36);
              v34 = (__int64 *)(v58 + 16LL * (unsigned int)v36);
              v33 = *v34;
              if ( v28 == *v34 )
                goto LABEL_62;
              ++v59;
            }
            goto LABEL_88;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 112);
        }
        sub_CE3D80(v78, 2 * v30);
        v48 = *(_DWORD *)(a1 + 136);
        if ( !v48 )
          goto LABEL_144;
        LODWORD(result) = ((unsigned int)v28 >> 4) ^ ((unsigned int)v28 >> 9);
        v49 = v48 - 1;
        v50 = *(_QWORD *)(a1 + 120);
        v32 = *(unsigned int *)(a1 + 128);
        v36 = v49 & (unsigned int)result;
        v35 = (unsigned int)(v32 + 1);
        v34 = (__int64 *)(v50 + 16 * v36);
        v33 = *v34;
        if ( v28 == *v34 )
          goto LABEL_62;
        v51 = 1;
        v32 = 0;
        while ( v33 != -4096 )
        {
          if ( !v32 && v33 == -8192 )
            v32 = (__int64)v34;
          v36 = v49 & (unsigned int)(v51 + v36);
          v34 = (__int64 *)(v50 + 16LL * (unsigned int)v36);
          v33 = *v34;
          if ( v28 == *v34 )
            goto LABEL_62;
          ++v51;
        }
LABEL_88:
        if ( v32 )
          v34 = (__int64 *)v32;
LABEL_62:
        *(_DWORD *)(a1 + 128) = v35;
        if ( *v34 != -4096 )
          --*(_DWORD *)(a1 + 132);
        *v34 = v28;
        v34[1] = 0;
        v37 = 0;
LABEL_30:
        v79 = result;
        sub_CE1BC0(v37 + 24, (__int64)v82, v32, v35, v36, v33);
        v38 = *(_DWORD *)(a1 + 136);
        if ( v38 )
        {
          v39 = *(_QWORD *)(a1 + 120);
          v40 = 1;
          v41 = (v38 - 1) & v79;
          v42 = 0;
          v43 = v39 + 16 * v41;
          v44 = *(_QWORD *)v43;
          if ( v28 == *(_QWORD *)v43 )
          {
LABEL_32:
            v45 = *(_QWORD *)(v43 + 8);
LABEL_33:
            v21 = (unsigned __int64)v82;
            result = sub_CE1BC0(v45 + 96, (__int64)v82, v40, v43, v44, v41);
            goto LABEL_20;
          }
          while ( v44 != -4096 )
          {
            if ( !v42 && v44 == -8192 )
              v42 = (__int64 *)v43;
            v41 = (v38 - 1) & ((_DWORD)v40 + (_DWORD)v41);
            v43 = v39 + 16LL * (unsigned int)v41;
            v44 = *(_QWORD *)v43;
            if ( v28 == *(_QWORD *)v43 )
              goto LABEL_32;
            v40 = (unsigned int)(v40 + 1);
          }
          v40 = *(unsigned int *)(a1 + 128);
          if ( !v42 )
            v42 = (__int64 *)v43;
          ++*(_QWORD *)(a1 + 112);
          v43 = (unsigned int)(v40 + 1);
          if ( 4 * (int)v43 < 3 * v38 )
          {
            v44 = v38 - *(_DWORD *)(a1 + 132) - (unsigned int)v43;
            v41 = v38 >> 3;
            if ( (unsigned int)v44 > (unsigned int)v41 )
              goto LABEL_49;
            sub_CE3D80(v78, v38);
            v60 = *(_DWORD *)(a1 + 136);
            if ( !v60 )
            {
LABEL_144:
              ++*(_DWORD *)(a1 + 128);
              BUG();
            }
            v61 = v60 - 1;
            v41 = *(_QWORD *)(a1 + 120);
            v55 = 0;
            v62 = v61 & v79;
            v43 = (unsigned int)(*(_DWORD *)(a1 + 128) + 1);
            v40 = 1;
            v42 = (__int64 *)(v41 + 16LL * (v61 & v79));
            v44 = *v42;
            if ( v28 == *v42 )
              goto LABEL_49;
            while ( v44 != -4096 )
            {
              if ( v44 == -8192 && !v55 )
                v55 = v42;
              v62 = v61 & (v40 + v62);
              v42 = (__int64 *)(v41 + 16LL * v62);
              v44 = *v42;
              if ( v28 == *v42 )
                goto LABEL_49;
              v40 = (unsigned int)(v40 + 1);
            }
            goto LABEL_94;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 112);
        }
        sub_CE3D80(v78, 2 * v38);
        v52 = *(_DWORD *)(a1 + 136);
        if ( !v52 )
          goto LABEL_144;
        v53 = v52 - 1;
        v41 = *(_QWORD *)(a1 + 120);
        v40 = *(unsigned int *)(a1 + 128);
        v54 = v53 & v79;
        v43 = (unsigned int)(v40 + 1);
        v42 = (__int64 *)(v41 + 16LL * (v53 & v79));
        v44 = *v42;
        if ( v28 == *v42 )
          goto LABEL_49;
        v40 = 1;
        v55 = 0;
        while ( v44 != -4096 )
        {
          if ( !v55 && v44 == -8192 )
            v55 = v42;
          v54 = v53 & (v40 + v54);
          v42 = (__int64 *)(v41 + 16LL * v54);
          v44 = *v42;
          if ( v28 == *v42 )
            goto LABEL_49;
          v40 = (unsigned int)(v40 + 1);
        }
LABEL_94:
        if ( v55 )
          v42 = v55;
LABEL_49:
        *(_DWORD *)(a1 + 128) = v43;
        if ( *v42 != -4096 )
          --*(_DWORD *)(a1 + 132);
        *v42 = v28;
        v42[1] = 0;
        v45 = 0;
        goto LABEL_33;
      }
LABEL_24:
      if ( (_BYTE *)v82[0] != v83 )
        return _libc_free(v82[0], v21);
    }
  }
  return result;
}
