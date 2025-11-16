// Function: sub_3203600
// Address: 0x3203600
//
__int64 __fastcall sub_3203600(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 *v10; // r14
  unsigned __int8 v11; // al
  unsigned int v12; // esi
  __int64 v13; // r15
  __int64 v14; // r9
  int v15; // r11d
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rdi
  int v28; // eax
  int v29; // eax
  __int64 v30; // r8
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rsi
  unsigned int v33; // edx
  __int64 v34; // rsi
  __int64 *v35; // rdi
  __int64 *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rsi
  unsigned __int64 *v39; // rsi
  int v40; // esi
  int v41; // esi
  unsigned int v42; // edx
  __int64 v43; // rdi
  int v44; // r10d
  _BYTE *v45; // rsi
  __int64 v46; // rax
  __int64 v47; // r8
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 v50; // rdx
  _BYTE *v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rdx
  _BOOL4 v54; // eax
  unsigned __int64 v55; // r15
  __int64 v56; // rdi
  int v57; // esi
  int v58; // esi
  int v59; // r10d
  unsigned int v60; // edx
  __int64 v61; // rdi
  unsigned __int64 v62; // [rsp+0h] [rbp-80h]
  __int64 v63; // [rsp+8h] [rbp-78h]
  __int64 v64; // [rsp+10h] [rbp-70h]
  __int64 v66; // [rsp+20h] [rbp-60h]
  __int64 v67; // [rsp+20h] [rbp-60h]
  unsigned __int64 v68; // [rsp+20h] [rbp-60h]
  unsigned __int64 v69; // [rsp+20h] [rbp-60h]
  __int64 v70; // [rsp+20h] [rbp-60h]
  unsigned int v71; // [rsp+20h] [rbp-60h]
  unsigned __int64 v72; // [rsp+20h] [rbp-60h]
  __int64 v73; // [rsp+38h] [rbp-48h] BYREF
  __int64 v74; // [rsp+40h] [rbp-40h] BYREF
  __int64 v75; // [rsp+48h] [rbp-38h]

  v64 = a1 + 48;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  v4 = *(_BYTE *)(a3 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a3 - 32);
  else
    v5 = a3 - 16 - 8LL * ((v4 >> 2) & 0xF);
  v6 = *(_QWORD *)(v5 + 32);
  if ( v6 )
  {
    v7 = *(_BYTE *)(v6 - 16);
    if ( (v7 & 2) != 0 )
    {
      v8 = *(__int64 **)(v6 - 32);
      v9 = *(unsigned int *)(v6 - 24);
    }
    else
    {
      v8 = (__int64 *)(v6 - 16 - 8LL * ((v7 >> 2) & 0xF));
      v9 = (*(_WORD *)(v6 - 16) >> 6) & 0xF;
    }
    v10 = &v8[v9];
    if ( v10 != v8 )
    {
      v63 = a1 + 104;
      while ( 1 )
      {
        v25 = *v8;
        if ( !*v8 )
          goto LABEL_17;
        if ( *(_BYTE *)v25 == 18 )
        {
          v11 = *(_BYTE *)(v25 - 16);
          if ( (v11 & 2) != 0 )
          {
            v12 = *(_DWORD *)(a1 + 72);
            v13 = *(_QWORD *)(*(_QWORD *)(v25 - 32) + 16LL);
            if ( v12 )
              goto LABEL_10;
LABEL_53:
            ++*(_QWORD *)(a1 + 48);
LABEL_54:
            sub_3203420(v64, 2 * v12);
            v40 = *(_DWORD *)(a1 + 72);
            if ( !v40 )
              goto LABEL_109;
            v41 = v40 - 1;
            v30 = *(_QWORD *)(a1 + 56);
            v42 = v41 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v29 = *(_DWORD *)(a1 + 64) + 1;
            v16 = v30 + 16LL * v42;
            v43 = *(_QWORD *)v16;
            if ( v13 != *(_QWORD *)v16 )
            {
              v44 = 1;
              v14 = 0;
              while ( v43 != -4096 )
              {
                if ( v43 == -8192 && !v14 )
                  v14 = v16;
                v42 = v41 & (v44 + v42);
                v16 = v30 + 16LL * v42;
                v43 = *(_QWORD *)v16;
                if ( v13 == *(_QWORD *)v16 )
                  goto LABEL_40;
                ++v44;
              }
LABEL_58:
              if ( v14 )
                v16 = v14;
            }
            goto LABEL_40;
          }
          v12 = *(_DWORD *)(a1 + 72);
          v13 = *(_QWORD *)(v25 - 8LL * ((v11 >> 2) & 0xF));
          if ( !v12 )
            goto LABEL_53;
LABEL_10:
          v14 = *(_QWORD *)(a1 + 56);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v18 = v14 + 16LL * v17;
          v19 = *(_QWORD *)v18;
          if ( v13 == *(_QWORD *)v18 )
          {
LABEL_11:
            v20 = *(unsigned int *)(v18 + 8);
          }
          else
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = (v12 - 1) & (v15 + v17);
              v18 = v14 + 16LL * v17;
              v19 = *(_QWORD *)v18;
              if ( v13 == *(_QWORD *)v18 )
                goto LABEL_11;
              ++v15;
            }
            if ( !v16 )
              v16 = v18;
            v28 = *(_DWORD *)(a1 + 64);
            ++*(_QWORD *)(a1 + 48);
            v29 = v28 + 1;
            if ( 4 * v29 >= 3 * v12 )
              goto LABEL_54;
            v30 = v12 >> 3;
            if ( v12 - *(_DWORD *)(a1 + 68) - v29 <= (unsigned int)v30 )
            {
              v71 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
              sub_3203420(v64, v12);
              v57 = *(_DWORD *)(a1 + 72);
              if ( !v57 )
              {
LABEL_109:
                ++*(_DWORD *)(a1 + 64);
                BUG();
              }
              v58 = v57 - 1;
              v30 = *(_QWORD *)(a1 + 56);
              v14 = 0;
              v59 = 1;
              v60 = v58 & v71;
              v29 = *(_DWORD *)(a1 + 64) + 1;
              v16 = v30 + 16LL * (v58 & v71);
              v61 = *(_QWORD *)v16;
              if ( *(_QWORD *)v16 != v13 )
              {
                while ( v61 != -4096 )
                {
                  if ( v61 == -8192 && !v14 )
                    v14 = v16;
                  v60 = v58 & (v59 + v60);
                  v16 = v30 + 16LL * v60;
                  v61 = *(_QWORD *)v16;
                  if ( v13 == *(_QWORD *)v16 )
                    goto LABEL_40;
                  ++v59;
                }
                goto LABEL_58;
              }
            }
LABEL_40:
            *(_DWORD *)(a1 + 64) = v29;
            if ( *(_QWORD *)v16 != -4096 )
              --*(_DWORD *)(a1 + 68);
            *(_QWORD *)v16 = v13;
            *(_DWORD *)(v16 + 8) = 0;
            v20 = *(unsigned int *)(a1 + 88);
            v31 = *(unsigned int *)(a1 + 92);
            v74 = v13;
            v32 = v20 + 1;
            v75 = 0;
            v33 = v20;
            if ( v20 + 1 > v31 )
            {
              v55 = *(_QWORD *)(a1 + 80);
              v70 = v16;
              v56 = a1 + 80;
              if ( v55 > (unsigned __int64)&v74 || (unsigned __int64)&v74 >= v55 + 16 * v20 )
              {
                sub_31FC930(v56, v32, (__int64)&v74, v16, v30, v14);
                v20 = *(unsigned int *)(a1 + 88);
                v34 = *(_QWORD *)(a1 + 80);
                v35 = &v74;
                v16 = v70;
                v33 = *(_DWORD *)(a1 + 88);
              }
              else
              {
                sub_31FC930(v56, v32, (__int64)&v74, v16, v30, v14);
                v34 = *(_QWORD *)(a1 + 80);
                v20 = *(unsigned int *)(a1 + 88);
                v16 = v70;
                v35 = (__int64 *)((char *)&v74 + v34 - v55);
                v33 = *(_DWORD *)(a1 + 88);
              }
            }
            else
            {
              v34 = *(_QWORD *)(a1 + 80);
              v35 = &v74;
            }
            v36 = (__int64 *)(16 * v20 + v34);
            if ( v36 )
            {
              *v36 = *v35;
              v37 = v35[1];
              v35[1] = 0;
              v36[1] = v37;
              v33 = *(_DWORD *)(a1 + 88);
              v38 = v75;
              *(_DWORD *)(a1 + 88) = v33 + 1;
              v20 = v33;
              if ( v38 )
              {
                if ( (v38 & 4) != 0 )
                {
                  v39 = (unsigned __int64 *)(v38 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( v39 )
                  {
                    if ( (unsigned __int64 *)*v39 != v39 + 2 )
                    {
                      v66 = v16;
                      _libc_free(*v39);
                      v16 = v66;
                    }
                    v67 = v16;
                    j_j___libc_free_0((unsigned __int64)v39);
                    v16 = v67;
                    v20 = (unsigned int)(*(_DWORD *)(a1 + 88) - 1);
                    v33 = *(_DWORD *)(a1 + 88) - 1;
                  }
                }
              }
            }
            else
            {
              *(_DWORD *)(a1 + 88) = v33 + 1;
            }
            *(_DWORD *)(v16 + 8) = v33;
          }
          v21 = *(_QWORD *)(a1 + 80) + 16 * v20;
          v22 = *(_QWORD *)(v21 + 8);
          v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v22 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v22 & 4) == 0 )
            {
              v68 = *(_QWORD *)(v21 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              v46 = sub_22077B0(0x30u);
              v47 = v68;
              if ( v46 )
              {
                *(_QWORD *)v46 = v46 + 16;
                *(_QWORD *)(v46 + 8) = 0x400000000LL;
              }
              v48 = v46;
              v49 = v46 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v21 + 8) = v48 | 4;
              v50 = *(unsigned int *)(v49 + 8);
              v14 = v50 + 1;
              if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(v49 + 12) )
              {
                v62 = v68;
                v72 = v49;
                sub_C8D5F0(v49, (const void *)(v49 + 16), v50 + 1, 8u, v47, v14);
                v49 = v72;
                v47 = v62;
                v50 = *(unsigned int *)(v72 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v49 + 8 * v50) = v47;
              ++*(_DWORD *)(v49 + 8);
              v23 = *(_QWORD *)(v21 + 8) & 0xFFFFFFFFFFFFFFF8LL;
            }
            v24 = *(unsigned int *)(v23 + 8);
            if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 12) )
            {
              v69 = v23;
              sub_C8D5F0(v23, (const void *)(v23 + 16), v24 + 1, 8u, v23, v14);
              v23 = v69;
              v24 = *(unsigned int *)(v69 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v23 + 8 * v24) = v25;
            ++*(_DWORD *)(v23 + 8);
          }
          else
          {
            *(_QWORD *)(v21 + 8) = v25 & 0xFFFFFFFFFFFFFFFBLL;
          }
          goto LABEL_17;
        }
        if ( *(_BYTE *)v25 == 13 )
        {
          v73 = *v8;
          if ( (unsigned __int16)sub_AF18C0(v73) == 13 )
          {
            sub_3203E00(a2, a1, v73);
            goto LABEL_17;
          }
          if ( (unsigned __int16)sub_AF18C0(v73) == 28 )
          {
            v51 = *(_BYTE **)(a1 + 8);
            if ( v51 == *(_BYTE **)(a1 + 16) )
            {
              sub_31FBF30(a1, v51, &v73);
            }
            else
            {
              if ( v51 )
              {
                *(_QWORD *)v51 = v73;
                v51 = *(_BYTE **)(a1 + 8);
              }
              *(_QWORD *)(a1 + 8) = v51 + 8;
            }
            goto LABEL_17;
          }
          if ( (unsigned __int16)sub_AF18C0(v73) == 15 && (v52 = sub_A547D0(v73, 2), v53 == 15) )
          {
            v54 = *(_QWORD *)v52 != 0x705F6C6274765F5FLL
               || *(_DWORD *)(v52 + 8) != 1952412276
               || *(_WORD *)(v52 + 12) != 28793
               || *(_BYTE *)(v52 + 14) != 101;
            v26 = v73;
            if ( !v54 )
            {
              *(_DWORD *)(a1 + 96) = sub_3206530(a2, v73, 0);
              goto LABEL_17;
            }
          }
          else
          {
            v26 = v73;
          }
          if ( (unsigned __int16)sub_AF18C0(v26) == 22 )
          {
            v45 = *(_BYTE **)(a1 + 112);
            v74 = v73;
            if ( v45 == *(_BYTE **)(a1 + 120) )
            {
LABEL_86:
              sub_31FC7A0(v63, v45, &v74);
              goto LABEL_17;
            }
            if ( v45 )
            {
              *(_QWORD *)v45 = v73;
              v45 = *(_BYTE **)(a1 + 112);
            }
            goto LABEL_65;
          }
          ++v8;
          sub_AF18C0(v73);
          if ( v10 == v8 )
            return a1;
        }
        else
        {
          v73 = 0;
          if ( *(_BYTE *)v25 == 14 )
          {
            v74 = v25;
            v45 = *(_BYTE **)(a1 + 112);
            if ( v45 == *(_BYTE **)(a1 + 120) )
              goto LABEL_86;
            if ( v45 )
            {
              *(_QWORD *)v45 = v25;
              v45 = *(_BYTE **)(a1 + 112);
            }
LABEL_65:
            *(_QWORD *)(a1 + 112) = v45 + 8;
          }
LABEL_17:
          if ( v10 == ++v8 )
            return a1;
        }
      }
    }
  }
  return a1;
}
