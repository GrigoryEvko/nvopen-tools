// Function: sub_37D7E90
// Address: 0x37d7e90
//
__int64 __fastcall sub_37D7E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 *v7; // rcx
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // r12
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 *v14; // rbx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // r15
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 *v21; // rbx
  __int64 v22; // r15
  __int64 *v23; // r14
  __int64 v24; // rsi
  bool v25; // zf
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  _BYTE *v28; // rdi
  unsigned int v29; // esi
  __int64 *v30; // rax
  int v31; // edx
  __int64 v32; // rcx
  _QWORD *v33; // r12
  __int64 *v34; // rbx
  __int64 v35; // rcx
  __int64 *v36; // rax
  __int64 *v37; // r12
  __int64 *v38; // rax
  __int64 v39; // rsi
  __int64 *v40; // rbx
  char v41; // di
  _QWORD *v42; // rax
  __int64 *v43; // rax
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  unsigned int v46; // esi
  __int64 v47; // r12
  unsigned int v48; // eax
  __int64 *v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 *v53; // rax
  int v54; // r10d
  _QWORD *v55; // rdx
  _QWORD *v56; // rcx
  unsigned int v57; // edx
  __int64 *v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rbx
  unsigned __int64 v61; // rdx
  __int64 *v62; // rax
  __int64 *v63; // r10
  int v64; // edx
  __int64 *v65; // rax
  unsigned int v66; // edi
  _QWORD *v67; // rax
  __int64 *v68; // rax
  int v69; // r11d
  __int64 *v70; // rcx
  int v71; // eax
  __int64 *v72; // r11
  __int64 v73; // [rsp+8h] [rbp-118h]
  __int64 *v74; // [rsp+10h] [rbp-110h]
  __int64 v75; // [rsp+10h] [rbp-110h]
  __int64 v76; // [rsp+18h] [rbp-108h]
  __int64 *v77; // [rsp+28h] [rbp-F8h]
  __int64 v78; // [rsp+30h] [rbp-F0h] BYREF
  __int64 *v79; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v80; // [rsp+40h] [rbp-E0h] BYREF
  __int64 *v81; // [rsp+48h] [rbp-D8h]
  __int64 v82; // [rsp+50h] [rbp-D0h]
  __int64 v83; // [rsp+58h] [rbp-C8h]
  _BYTE *v84; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v85; // [rsp+68h] [rbp-B8h]
  _BYTE v86[176]; // [rsp+70h] [rbp-B0h] BYREF

  v4 = a1;
  sub_35059C0((_QWORD *)(a1 + 128), a2, a3);
  v9 = *(__int64 **)(a4 + 8);
  if ( *(_BYTE *)(a4 + 28) )
    v10 = *(unsigned int *)(a4 + 20);
  else
    v10 = *(unsigned int *)(a4 + 16);
  v11 = &v9[v10];
  v12 = *(unsigned __int8 *)(a3 + 28);
  if ( v9 != v11 )
  {
    while ( 1 )
    {
      v13 = *v9;
      v14 = v9;
      if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v11 == ++v9 )
        goto LABEL_6;
    }
    if ( v11 != v9 )
    {
      if ( !(_BYTE)v12 )
        goto LABEL_109;
      while ( 1 )
      {
        v10 = *(_QWORD *)(a3 + 8);
        v7 = (__int64 *)(v10 + 8LL * *(unsigned int *)(a3 + 20));
        v66 = *(_DWORD *)(a3 + 20);
        if ( (__int64 *)v10 != v7 )
          break;
LABEL_110:
        if ( v66 < *(_DWORD *)(a3 + 16) )
        {
          *(_DWORD *)(a3 + 20) = v66 + 1;
          *v7 = v13;
          v10 = *(_QWORD *)(a3 + 8);
          ++*(_QWORD *)a3;
          v12 = *(unsigned __int8 *)(a3 + 28);
          goto LABEL_103;
        }
        do
        {
LABEL_109:
          sub_C8CC70(a3, v13, v10, (__int64)v7, v12, v8);
          v12 = *(unsigned __int8 *)(a3 + 28);
          v10 = *(_QWORD *)(a3 + 8);
LABEL_103:
          v68 = v14 + 1;
          if ( v14 + 1 != v11 )
          {
            while ( 1 )
            {
              v13 = *v68;
              v14 = v68;
              if ( (unsigned __int64)*v68 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v11 == ++v68 )
                goto LABEL_7;
            }
            if ( v11 != v68 )
              continue;
          }
          goto LABEL_7;
        }
        while ( !(_BYTE)v12 );
      }
      v67 = *(_QWORD **)(a3 + 8);
      while ( *v67 != v13 )
      {
        if ( v7 == ++v67 )
          goto LABEL_110;
      }
      goto LABEL_103;
    }
  }
LABEL_6:
  v10 = *(_QWORD *)(a3 + 8);
LABEL_7:
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  if ( (_BYTE)v12 )
    v77 = (__int64 *)(v10 + 8LL * *(unsigned int *)(a3 + 20));
  else
    v77 = (__int64 *)(v10 + 8LL * *(unsigned int *)(a3 + 16));
  v15 = (__int64 *)v10;
  if ( v77 == (__int64 *)v10 )
    goto LABEL_12;
  while ( 1 )
  {
    v16 = *v15;
    v17 = v15;
    if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v77 == ++v15 )
      goto LABEL_12;
  }
  if ( v77 == v15 )
  {
LABEL_12:
    v18 = 0;
    v19 = 0;
    return sub_C7D6A0(v18, v19, 8);
  }
  v76 = v4 + 440;
  do
  {
    v21 = *(__int64 **)(v16 + 112);
    v84 = v86;
    v85 = 0x800000000LL;
    if ( v21 != &v21[*(unsigned int *)(v16 + 120)] )
    {
      v74 = v17;
      v22 = v4;
      v23 = &v21[*(unsigned int *)(v16 + 120)];
      while ( 1 )
      {
        v24 = *v21;
        v25 = *(_BYTE *)(a3 + 28) == 0;
        v78 = *v21;
        if ( v25 )
        {
          if ( sub_C8CA60(a3, v24) )
            goto LABEL_23;
          v24 = v78;
        }
        else
        {
          v26 = *(_QWORD **)(a3 + 8);
          v27 = &v26[*(unsigned int *)(a3 + 20)];
          if ( v26 != v27 )
          {
            while ( v24 != *v26 )
            {
              if ( v27 == ++v26 )
                goto LABEL_56;
            }
            goto LABEL_23;
          }
        }
LABEL_56:
        if ( *(_BYTE *)(v22 + 468) )
        {
          v44 = *(_QWORD **)(v22 + 448);
          v45 = &v44[*(unsigned int *)(v22 + 460)];
          if ( v44 == v45 )
            goto LABEL_23;
          while ( *v44 != v24 )
          {
            if ( v45 == ++v44 )
              goto LABEL_23;
          }
          v46 = v83;
          if ( !(_DWORD)v83 )
          {
LABEL_81:
            ++v80;
            v79 = 0;
            goto LABEL_82;
          }
LABEL_62:
          v47 = v78;
          v48 = (v46 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
          v49 = &v81[v48];
          v50 = *v49;
          if ( *v49 != v78 )
          {
            v12 = 1;
            v63 = 0;
            while ( v50 != -4096 )
            {
              if ( v50 != -8192 || v63 )
                v49 = v63;
              v48 = (v46 - 1) & (v12 + v48);
              v72 = &v81[v48];
              v50 = *v72;
              if ( v78 == *v72 )
                goto LABEL_63;
              v12 = (unsigned int)(v12 + 1);
              v63 = v49;
              v49 = &v81[v48];
            }
            if ( !v63 )
              v63 = v49;
            ++v80;
            v64 = v82 + 1;
            v79 = v63;
            if ( 4 * ((int)v82 + 1) >= 3 * v46 )
            {
LABEL_82:
              v46 *= 2;
            }
            else if ( v46 - HIDWORD(v82) - v64 > v46 >> 3 )
            {
              goto LABEL_94;
            }
            sub_2E61F50((__int64)&v80, v46);
            sub_37C00E0((__int64)&v80, &v78, &v79);
            v47 = v78;
            v63 = v79;
            v64 = v82 + 1;
LABEL_94:
            LODWORD(v82) = v64;
            if ( *v63 != -4096 )
              --HIDWORD(v82);
            *v63 = v47;
            v47 = v78;
          }
LABEL_63:
          v51 = (unsigned int)v85;
          v8 = *(_QWORD *)(v47 + 112);
          v52 = (unsigned int)v85 + 1LL;
          if ( v52 > HIDWORD(v85) )
          {
            v73 = *(_QWORD *)(v47 + 112);
            sub_C8D5F0((__int64)&v84, v86, v52, 0x10u, v12, v8);
            v51 = (unsigned int)v85;
            v8 = v73;
          }
          ++v21;
          v53 = (__int64 *)&v84[16 * v51];
          *v53 = v47;
          v53[1] = v8;
          LODWORD(v85) = v85 + 1;
          if ( v23 == v21 )
          {
LABEL_24:
            v16 = (unsigned int)v85;
            v4 = v22;
            v28 = v84;
            v17 = v74;
            if ( !(_DWORD)v85 )
            {
LABEL_32:
              if ( v28 != v86 )
                _libc_free((unsigned __int64)v28);
              break;
            }
            while ( 2 )
            {
              while ( 1 )
              {
                v33 = &v28[16 * (unsigned int)v16 - 16];
                v34 = (__int64 *)v33[1];
                if ( v34 != (__int64 *)(*(_QWORD *)(*v33 + 112LL) + 8LL * *(unsigned int *)(*v33 + 120LL)) )
                  break;
                v16 = (unsigned int)(v16 - 1);
                LODWORD(v85) = v16;
                if ( !(_DWORD)v16 )
                  goto LABEL_32;
              }
              v29 = v83;
              v12 = *v34;
              v30 = v81;
              if ( (_DWORD)v83 )
              {
                v8 = (unsigned int)(v83 - 1);
                v31 = v8 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
                v32 = v81[v31];
                if ( v12 == v32 )
                  goto LABEL_28;
                v54 = 1;
                while ( v32 != -4096 )
                {
                  v31 = v8 & (v54 + v31);
                  v32 = v81[v31];
                  if ( v12 == v32 )
                    goto LABEL_28;
                  ++v54;
                }
              }
              if ( *(_BYTE *)(v4 + 468) )
              {
                v55 = *(_QWORD **)(v4 + 448);
                v56 = &v55[*(unsigned int *)(v4 + 460)];
                if ( v55 == v56 )
                {
LABEL_28:
                  v33[1] = v34 + 1;
                  v16 = (unsigned int)v85;
                  goto LABEL_29;
                }
                while ( v12 != *v55 )
                {
                  if ( v56 == ++v55 )
                    goto LABEL_28;
                }
              }
              else
              {
                v65 = sub_C8CA60(v76, *v34);
                v34 = (__int64 *)v33[1];
                if ( !v65 )
                {
                  v28 = v84;
                  goto LABEL_28;
                }
                v30 = v81;
                v29 = v83;
              }
              if ( v29 )
              {
                v12 = *v34;
                v57 = (v29 - 1) & (((unsigned int)*v34 >> 9) ^ ((unsigned int)*v34 >> 4));
                v58 = &v30[v57];
                v8 = *v58;
                if ( *v58 == *v34 )
                {
LABEL_76:
                  v59 = (unsigned int)v85;
                  v60 = *(_QWORD *)(v12 + 112);
                  v61 = (unsigned int)v85 + 1LL;
                  if ( v61 > HIDWORD(v85) )
                  {
                    v75 = v12;
                    sub_C8D5F0((__int64)&v84, v86, v61, 0x10u, v12, v8);
                    v59 = (unsigned int)v85;
                    v12 = v75;
                  }
                  v62 = (__int64 *)&v84[16 * v59];
                  *v62 = v12;
                  v28 = v84;
                  v62[1] = v60;
                  v16 = (unsigned int)(v85 + 1);
                  LODWORD(v85) = v85 + 1;
LABEL_29:
                  if ( !(_DWORD)v16 )
                    goto LABEL_32;
                  continue;
                }
                v69 = 1;
                v70 = 0;
                while ( v8 != -4096 )
                {
                  if ( v8 == -8192 && !v70 )
                    v70 = v58;
                  v57 = (v29 - 1) & (v69 + v57);
                  v58 = &v30[v57];
                  v8 = *v58;
                  if ( v12 == *v58 )
                    goto LABEL_76;
                  ++v69;
                }
                if ( !v70 )
                  v70 = v58;
                ++v80;
                v71 = v82 + 1;
                v79 = v70;
                if ( 4 * ((int)v82 + 1) < 3 * v29 )
                {
                  if ( v29 - (v71 + HIDWORD(v82)) > v29 >> 3 )
                  {
LABEL_127:
                    LODWORD(v82) = v71;
                    if ( *v70 != -4096 )
                      --HIDWORD(v82);
                    *v70 = *v34;
                    v12 = *(_QWORD *)v33[1];
                    goto LABEL_76;
                  }
LABEL_132:
                  sub_2E61F50((__int64)&v80, v29);
                  sub_37C00E0((__int64)&v80, v34, &v79);
                  v70 = v79;
                  v71 = v82 + 1;
                  goto LABEL_127;
                }
              }
              else
              {
                ++v80;
                v79 = 0;
              }
              break;
            }
            v29 *= 2;
            goto LABEL_132;
          }
        }
        else
        {
          if ( sub_C8CA60(v76, v24) )
          {
            v46 = v83;
            if ( !(_DWORD)v83 )
              goto LABEL_81;
            goto LABEL_62;
          }
LABEL_23:
          if ( v23 == ++v21 )
            goto LABEL_24;
        }
      }
    }
    v35 = (__int64)v77;
    v36 = v17 + 1;
    if ( v17 + 1 == v77 )
      break;
    while ( 1 )
    {
      v16 = *v36;
      v17 = v36;
      if ( (unsigned __int64)*v36 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v77 == ++v36 )
        goto LABEL_37;
    }
  }
  while ( v77 != v36 );
LABEL_37:
  v18 = (__int64)v81;
  v19 = 8LL * (unsigned int)v83;
  v37 = &v81[(unsigned __int64)v19 / 8];
  if ( (_DWORD)v82 && v81 != v37 )
  {
    v38 = v81;
    while ( 1 )
    {
      v39 = *v38;
      v40 = v38;
      if ( *v38 != -8192 && v39 != -4096 )
        break;
      if ( v37 == ++v38 )
        return sub_C7D6A0(v18, v19, 8);
    }
    if ( v38 != v37 )
    {
      v41 = *(_BYTE *)(a3 + 28);
      if ( !v41 )
        goto LABEL_118;
LABEL_45:
      v42 = *(_QWORD **)(a3 + 8);
      v35 = *(unsigned int *)(a3 + 20);
      v16 = (__int64)&v42[v35];
      if ( v42 == (_QWORD *)v16 )
      {
LABEL_119:
        if ( (unsigned int)v35 < *(_DWORD *)(a3 + 16) )
        {
          v35 = (unsigned int)(v35 + 1);
          *(_DWORD *)(a3 + 20) = v35;
          *(_QWORD *)v16 = v39;
          v41 = *(_BYTE *)(a3 + 28);
          ++*(_QWORD *)a3;
          goto LABEL_49;
        }
        goto LABEL_118;
      }
      while ( v39 != *v42 )
      {
        if ( (_QWORD *)v16 == ++v42 )
          goto LABEL_119;
      }
LABEL_49:
      while ( 1 )
      {
        v43 = v40 + 1;
        if ( v40 + 1 == v37 )
          break;
        while ( 1 )
        {
          v39 = *v43;
          v40 = v43;
          if ( *v43 != -8192 && v39 != -4096 )
            break;
          if ( v37 == ++v43 )
            goto LABEL_53;
        }
        if ( v43 == v37 )
          break;
        if ( v41 )
          goto LABEL_45;
LABEL_118:
        sub_C8CC70(a3, v39, v16, v35, v19, v8);
        v41 = *(_BYTE *)(a3 + 28);
      }
LABEL_53:
      v18 = (__int64)v81;
      v19 = 8LL * (unsigned int)v83;
    }
  }
  return sub_C7D6A0(v18, v19, 8);
}
