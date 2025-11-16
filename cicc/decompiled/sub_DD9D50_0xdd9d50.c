// Function: sub_DD9D50
// Address: 0xdd9d50
//
__int64 __fastcall sub_DD9D50(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  __int64 v7; // r15
  __int64 v8; // r14
  unsigned __int64 v9; // r12
  unsigned int v10; // ebx
  char v12; // dl
  __int16 v13; // ax
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // r8
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  __int64 *v25; // r13
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // rcx
  int v29; // edx
  unsigned int v30; // esi
  __int64 *v31; // rax
  __int64 v32; // r9
  __int64 v33; // rdx
  char v34; // al
  char v35; // dl
  _QWORD *v36; // rax
  _QWORD *v37; // rcx
  int v38; // esi
  _QWORD *v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned int v44; // edx
  __int64 v45; // rax
  int v46; // eax
  int v47; // edi
  __int64 v48; // rcx
  int v49; // edx
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // r13
  unsigned __int64 v54; // r12
  __int64 v55; // rcx
  int v56; // edx
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 *v60; // r13
  _QWORD *v61; // r15
  __int64 *v62; // rax
  __int64 *v63; // rax
  unsigned int v64; // edx
  unsigned int v65; // esi
  __int64 v66; // r15
  __int64 v67; // rdx
  __int64 v68; // r14
  __int64 v69; // r15
  __int64 v70; // r8
  __int64 v71; // rax
  int v72; // ecx
  __int64 v73; // rax
  __int64 *v74; // r13
  int v75; // ecx
  __int64 v76; // r8
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 *v79; // r15
  __int64 v80; // [rsp+8h] [rbp-78h]
  __int64 v81; // [rsp+10h] [rbp-70h]
  unsigned int v84; // [rsp+28h] [rbp-58h]
  unsigned int v85; // [rsp+2Ch] [rbp-54h]
  __int64 v86; // [rsp+30h] [rbp-50h]
  __int64 v87; // [rsp+38h] [rbp-48h]
  __int64 v89; // [rsp+48h] [rbp-38h]
  __int64 *v90; // [rsp+48h] [rbp-38h]

  v7 = a4;
  v8 = a1;
  v9 = a2;
  v84 = a2;
  if ( *(_WORD *)(a3 + 24) == 15 )
  {
    v86 = *(_QWORD *)(a3 - 8);
    if ( *(_BYTE *)v86 == 84 )
    {
      v80 = a1 + 512;
      sub_AE6EC0(a1 + 512, v86);
      if ( !v12 )
        return 0;
      v13 = *(_WORD *)(v7 + 24);
      v14 = BYTE4(a2);
      if ( v13 != 15 )
      {
        v89 = *(_QWORD *)(v86 + 40);
        if ( v13 != 8 )
        {
LABEL_48:
          v81 = 0;
          goto LABEL_12;
        }
        goto LABEL_10;
      }
      v81 = *(_QWORD *)(v7 - 8);
      if ( *(_BYTE *)v81 != 84 )
      {
        v81 = 0;
        v89 = *(_QWORD *)(v86 + 40);
        goto LABEL_12;
      }
      sub_AE6EC0(v80, v81);
      v14 = BYTE4(a2);
      v65 = v64;
      v35 = *(_BYTE *)(a1 + 540);
      v10 = v65;
      v34 = v35;
      if ( !(_BYTE)v65 )
      {
        v81 = 0;
        goto LABEL_33;
      }
      v89 = *(_QWORD *)(v86 + 40);
      if ( *(_WORD *)(v7 + 24) == 8 )
      {
        if ( v89 != *(_QWORD *)(v81 + 40) )
          goto LABEL_11;
      }
      else if ( v89 != *(_QWORD *)(v81 + 40) )
      {
LABEL_12:
        v16 = *(_QWORD *)(v89 + 16);
        if ( v16 )
        {
          while ( 1 )
          {
            v17 = *(_QWORD *)(v16 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
              break;
            v16 = *(_QWORD *)(v16 + 8);
            if ( !v16 )
            {
              v34 = *(_BYTE *)(a1 + 540);
              v10 = 1;
              goto LABEL_32;
            }
          }
          v87 = v14 << 32;
          v18 = v16;
          v85 = ((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4);
          v19 = (__int64 *)a1;
          while ( 1 )
          {
            v20 = *(_QWORD *)(v17 + 40);
            if ( !sub_DAEB40((__int64)v19, v7, v20) )
              break;
            v21 = *(_QWORD *)(v86 - 8);
            v22 = 0x1FFFFFFFE0LL;
            v23 = *(_DWORD *)(v86 + 4) & 0x7FFFFFF;
            if ( v23 )
            {
              v24 = 0;
              do
              {
                if ( v20 == *(_QWORD *)(v21 + 32LL * *(unsigned int *)(v86 + 72) + 8 * v24) )
                {
                  v22 = 32 * v24;
                  goto LABEL_21;
                }
                ++v24;
              }
              while ( v23 != (_DWORD)v24 );
              v22 = 0x1FFFFFFFE0LL;
            }
LABEL_21:
            v25 = sub_DD8400((__int64)v19, *(_QWORD *)(v21 + v22));
            if ( !sub_DAEB50((__int64)v19, (__int64)v25, v89) )
              break;
            v26 = v19[6];
            v27 = *(_DWORD *)(v26 + 24);
            v28 = *(_QWORD *)(v26 + 8);
            if ( v27 )
            {
              v29 = v27 - 1;
              v30 = v29 & v85;
              v31 = (__int64 *)(v28 + 16LL * (v29 & v85));
              v32 = *v31;
              if ( *v31 == v89 )
              {
LABEL_24:
                v33 = v31[1];
                if ( v33 && sub_DAE0A0((__int64)v19, (__int64)v25, v33) )
                  break;
              }
              else
              {
                v46 = 1;
                while ( v32 != -4096 )
                {
                  v47 = v46 + 1;
                  v30 = v29 & (v46 + v30);
                  v31 = (__int64 *)(v28 + 16LL * v30);
                  v32 = *v31;
                  if ( *v31 == v89 )
                    goto LABEL_24;
                  v46 = v47;
                }
              }
            }
            v9 = v87 | v84 | v9 & 0xFFFFFF0000000000LL;
            if ( !(unsigned __int8)sub_DCD020(v19, v9, (__int64)v25, v7)
              && !(unsigned __int8)sub_DC15B0((__int64)v19, v9, (__int64)v25, v7, v9, a5, a6)
              && !(unsigned __int8)sub_DDA790((_DWORD)v19, v9, (_DWORD)v25, v7, a5, a6, a7) )
            {
              break;
            }
            v18 = *(_QWORD *)(v18 + 8);
            if ( !v18 )
            {
LABEL_30:
              v8 = (__int64)v19;
              goto LABEL_31;
            }
            while ( 1 )
            {
              v17 = *(_QWORD *)(v18 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
                break;
              v18 = *(_QWORD *)(v18 + 8);
              if ( !v18 )
                goto LABEL_30;
            }
          }
LABEL_54:
          v8 = (__int64)v19;
LABEL_55:
          v34 = *(_BYTE *)(v8 + 540);
          v10 = 0;
          goto LABEL_32;
        }
LABEL_31:
        v34 = *(_BYTE *)(v8 + 540);
        v10 = 1;
LABEL_32:
        v35 = v34;
        goto LABEL_33;
      }
      v66 = *(_QWORD *)(v89 + 16);
      if ( !v66 )
        goto LABEL_32;
      while ( 1 )
      {
        v67 = *(_QWORD *)(v66 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v67 - 30) <= 0xAu )
          break;
        v66 = *(_QWORD *)(v66 + 8);
        if ( !v66 )
          goto LABEL_32;
      }
      v19 = (__int64 *)a1;
      v68 = v66;
      while ( 1 )
      {
        v69 = *(_QWORD *)(v67 + 40);
        v70 = *(_QWORD *)(v86 - 8);
        v71 = 0x1FFFFFFFE0LL;
        v72 = *(_DWORD *)(v86 + 4) & 0x7FFFFFF;
        if ( v72 )
        {
          v73 = 0;
          do
          {
            if ( v69 == *(_QWORD *)(v70 + 32LL * *(unsigned int *)(v86 + 72) + 8 * v73) )
            {
              v71 = 32 * v73;
              goto LABEL_102;
            }
            ++v73;
          }
          while ( v72 != (_DWORD)v73 );
          v71 = 0x1FFFFFFFE0LL;
        }
LABEL_102:
        v74 = sub_DD8400(a1, *(_QWORD *)(v70 + v71));
        v75 = *(_DWORD *)(v81 + 4) & 0x7FFFFFF;
        if ( v75 )
        {
          v76 = *(_QWORD *)(v81 - 8);
          v77 = 0;
          do
          {
            if ( v69 == *(_QWORD *)(v76 + 32LL * *(unsigned int *)(v81 + 72) + 8 * v77) )
            {
              v78 = 32 * v77;
              goto LABEL_107;
            }
            ++v77;
          }
          while ( v75 != (_DWORD)v77 );
          v78 = 0x1FFFFFFFE0LL;
        }
        else
        {
          v76 = *(_QWORD *)(v81 - 8);
          v78 = 0x1FFFFFFFE0LL;
        }
LABEL_107:
        v79 = sub_DD8400(a1, *(_QWORD *)(v76 + v78));
        v9 = ((unsigned __int64)BYTE4(a2) << 32) | v84 | v9 & 0xFFFFFF0000000000LL;
        if ( !(unsigned __int8)sub_DCD020((__int64 *)a1, v9, (__int64)v74, (__int64)v79)
          && !(unsigned __int8)sub_DC15B0(a1, v9, (__int64)v74, (__int64)v79, v9, a5, a6)
          && !(unsigned __int8)sub_DDA790(a1, v9, (_DWORD)v74, (_DWORD)v79, a5, a6, a7) )
        {
          goto LABEL_54;
        }
        v68 = *(_QWORD *)(v68 + 8);
        if ( !v68 )
        {
LABEL_111:
          v8 = a1;
          v10 = (unsigned __int8)v65;
          v34 = *(_BYTE *)(a1 + 540);
          goto LABEL_32;
        }
        while ( 1 )
        {
          v67 = *(_QWORD *)(v68 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v67 - 30) <= 0xAu )
            break;
          v68 = *(_QWORD *)(v68 + 8);
          if ( !v68 )
            goto LABEL_111;
        }
      }
    }
  }
  if ( *(_WORD *)(a4 + 24) != 15 )
    return 0;
  v86 = *(_QWORD *)(a4 - 8);
  if ( *(_BYTE *)v86 != 84 )
    return 0;
  v80 = a1 + 512;
  sub_AE6EC0(a1 + 512, v86);
  v10 = v44;
  if ( !(_BYTE)v44 )
    return v10;
  v84 = sub_B52F50(a2);
  v14 = BYTE4(a2);
  v7 = a3;
  v89 = *(_QWORD *)(v86 + 40);
  v45 = a6;
  a6 = a5;
  a5 = v45;
  if ( *(_WORD *)(a3 + 24) != 8 )
    goto LABEL_48;
LABEL_10:
  v81 = 0;
LABEL_11:
  v15 = *(_QWORD *)(v7 + 48);
  if ( **(_QWORD **)(v15 + 32) != v89 )
    goto LABEL_12;
  if ( (*(_DWORD *)(v86 + 4) & 0x7FFFFFF) != 2 )
    goto LABEL_55;
  v48 = sub_D47840(v15);
  v49 = *(_DWORD *)(v86 + 4) & 0x7FFFFFF;
  if ( v49 )
  {
    v50 = *(_QWORD *)(v86 - 8);
    v51 = 0;
    do
    {
      if ( v48 == *(_QWORD *)(v50 + 32LL * *(unsigned int *)(v86 + 72) + 8 * v51) )
      {
        v52 = 32 * v51;
        goto LABEL_66;
      }
      ++v51;
    }
    while ( v49 != (_DWORD)v51 );
    v52 = 0x1FFFFFFFE0LL;
  }
  else
  {
    v50 = *(_QWORD *)(v86 - 8);
    v52 = 0x1FFFFFFFE0LL;
  }
LABEL_66:
  v90 = sub_DD8400(v8, *(_QWORD *)(v50 + v52));
  v53 = **(_QWORD **)(v7 + 32);
  v54 = v84 | v9 & 0xFFFFFFFF00000000LL;
  if ( !(unsigned __int8)sub_DCD020((__int64 *)v8, v54, (__int64)v90, v53)
    && !(unsigned __int8)sub_DC15B0(v8, v54, (__int64)v90, v53, v54, a5, a6)
    && !(unsigned __int8)sub_DDA790(v8, v54, (_DWORD)v90, v53, a5, a6, a7) )
  {
    goto LABEL_55;
  }
  v55 = sub_D47930(v15);
  v56 = *(_DWORD *)(v86 + 4) & 0x7FFFFFF;
  if ( v56 )
  {
    v57 = *(_QWORD *)(v86 - 8);
    v58 = 0;
    do
    {
      if ( v55 == *(_QWORD *)(v57 + 32LL * *(unsigned int *)(v86 + 72) + 8 * v58) )
      {
        v59 = 32 * v58;
        goto LABEL_72;
      }
      ++v58;
    }
    while ( v56 != (_DWORD)v58 );
    v59 = 0x1FFFFFFFE0LL;
LABEL_72:
    v60 = sub_DD8400(v8, *(_QWORD *)(v57 + v59));
    v61 = sub_DCC620(v7, (__int64 *)v8);
    if ( (unsigned __int8)sub_DCD020((__int64 *)v8, v54, (__int64)v60, (__int64)v61) )
      goto LABEL_31;
LABEL_73:
    if ( !(unsigned __int8)sub_DC15B0(v8, v54, (__int64)v60, (__int64)v61, v54, a5, a6) )
    {
      v10 = sub_DDA790(v8, v54, (_DWORD)v60, (_DWORD)v61, a5, a6, a7);
      v34 = *(_BYTE *)(v8 + 540);
      goto LABEL_32;
    }
    goto LABEL_31;
  }
  v60 = sub_DD8400(v8, *(_QWORD *)(*(_QWORD *)(v86 - 8) + 0x1FFFFFFFE0LL));
  v61 = sub_DCC620(v7, (__int64 *)v8);
  v10 = sub_DCD020((__int64 *)v8, v54, (__int64)v60, (__int64)v61);
  if ( !(_BYTE)v10 )
    goto LABEL_73;
  v35 = *(_BYTE *)(v8 + 540);
LABEL_33:
  if ( v35 )
  {
    v36 = *(_QWORD **)(v8 + 520);
    v37 = &v36[*(unsigned int *)(v8 + 532)];
    v38 = *(_DWORD *)(v8 + 532);
    if ( v36 != v37 )
    {
      v39 = *(_QWORD **)(v8 + 520);
      do
      {
        if ( *v39 == v86 )
        {
          v40 = (unsigned int)(v38 - 1);
          *(_DWORD *)(v8 + 532) = v40;
          *v39 = v36[v40];
          ++*(_QWORD *)(v8 + 512);
          goto LABEL_39;
        }
        ++v39;
      }
      while ( v37 != v39 );
      v42 = *(_QWORD *)(v8 + 520);
      if ( v81 )
      {
        v41 = v81;
LABEL_44:
        while ( v41 != *v36 )
        {
          if ( v37 == ++v36 )
            return v10;
        }
        v43 = (unsigned int)(v38 - 1);
        *(_DWORD *)(v8 + 532) = v43;
        *v36 = *(_QWORD *)(v42 + 8 * v43);
        ++*(_QWORD *)(v8 + 512);
      }
    }
  }
  else
  {
    v63 = sub_C8CA60(v80, v86);
    if ( v63 )
    {
      *v63 = -2;
      ++*(_DWORD *)(v8 + 536);
      ++*(_QWORD *)(v8 + 512);
    }
LABEL_39:
    v41 = v81;
    if ( v81 )
    {
      if ( *(_BYTE *)(v8 + 540) )
      {
        v36 = *(_QWORD **)(v8 + 520);
        v37 = &v36[*(unsigned int *)(v8 + 532)];
        v38 = *(_DWORD *)(v8 + 532);
        if ( v36 != v37 )
        {
          v42 = *(_QWORD *)(v8 + 520);
          goto LABEL_44;
        }
      }
      else
      {
        v62 = sub_C8CA60(v80, v81);
        if ( v62 )
        {
          *v62 = -2;
          ++*(_DWORD *)(v8 + 536);
          ++*(_QWORD *)(v8 + 512);
        }
      }
    }
  }
  return v10;
}
