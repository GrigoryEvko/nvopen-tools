// Function: sub_30D9A60
// Address: 0x30d9a60
//
__int64 __fastcall sub_30D9A60(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // r13
  __int64 *v5; // r14
  __int64 *v6; // rsi
  __int64 *v7; // rdx
  int v8; // eax
  int v9; // edi
  __int64 v10; // r15
  int v11; // ecx
  unsigned int v12; // r8d
  unsigned __int8 v13; // al
  bool v14; // al
  __int64 *v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rdx
  unsigned int v18; // r13d
  unsigned __int64 v20; // rax
  unsigned __int8 v21; // cl
  int v22; // ecx
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rsi
  unsigned __int8 v27; // al
  __int64 v28; // rax
  unsigned int v29; // edx
  unsigned int v30; // r13d
  int v31; // eax
  __int64 v32; // rcx
  int v33; // eax
  unsigned int v34; // esi
  __int64 *v35; // rdx
  __int64 v36; // rdi
  unsigned int v37; // edi
  __int64 **v38; // rdx
  __int64 *v39; // r8
  int v40; // eax
  int v41; // eax
  int v42; // ecx
  char v43; // al
  unsigned __int8 *v44; // rbx
  unsigned __int8 *v45; // rbx
  __int64 v46; // rdx
  unsigned __int8 *v47; // rbx
  bool v48; // al
  __int64 v49; // r14
  int v50; // edx
  unsigned __int8 v51; // al
  unsigned __int8 v52; // al
  int v53; // edx
  int v54; // edx
  int v55; // r8d
  int v56; // r9d
  unsigned __int8 *v57; // rbx
  unsigned __int8 *v58; // rbx
  __int64 v59; // rdx
  unsigned __int8 v60; // al
  unsigned __int8 *v61; // rbx
  __int64 v62; // r15
  void **v63; // rax
  void **v64; // r14
  char v65; // al
  void **v66; // r14
  __int64 v67; // r15
  void **v68; // rax
  void **v69; // r14
  void **v70; // r14
  int v71; // eax
  char v72; // r15
  _BYTE *v73; // rax
  _BYTE *v74; // r14
  char v75; // al
  _BYTE *v76; // r14
  int v77; // eax
  void **v78; // rax
  void **v79; // r14
  char v80; // al
  _BYTE *v81; // r14
  __int64 *v82; // [rsp+8h] [rbp-88h]
  int v83; // [rsp+8h] [rbp-88h]
  int v84; // [rsp+8h] [rbp-88h]
  __m128i v85; // [rsp+10h] [rbp-80h] BYREF
  __int64 v86; // [rsp+20h] [rbp-70h]
  __int64 v87; // [rsp+28h] [rbp-68h]
  __int64 v88; // [rsp+30h] [rbp-60h]
  __int64 v89; // [rsp+38h] [rbp-58h]
  __int64 v90; // [rsp+40h] [rbp-50h]
  __int64 v91; // [rsp+48h] [rbp-48h]
  __int16 v92; // [rsp+50h] [rbp-40h]

  v5 = (__int64 *)*((_QWORD *)a2 - 4);
  v6 = (__int64 *)*((_QWORD *)a2 - 8);
  v4 = (__int64)v6;
  if ( *(_BYTE *)v6 <= 0x15u )
    goto LABEL_2;
  v31 = *(_DWORD *)(a1 + 160);
  v32 = *(_QWORD *)(a1 + 144);
  if ( !v31 )
  {
    v6 = 0;
LABEL_2:
    v7 = v5;
    if ( *(_BYTE *)v5 <= 0x15u )
      goto LABEL_3;
    v40 = *(_DWORD *)(a1 + 160);
    v32 = *(_QWORD *)(a1 + 144);
    v7 = 0;
    if ( !v40 )
      goto LABEL_3;
    v33 = v40 - 1;
    goto LABEL_47;
  }
  v33 = v31 - 1;
  v34 = v33 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v35 = (__int64 *)(v32 + 16LL * v34);
  v36 = *v35;
  if ( v4 == *v35 )
  {
LABEL_45:
    v6 = (__int64 *)v35[1];
  }
  else
  {
    v53 = 1;
    while ( v36 != -4096 )
    {
      v55 = v53 + 1;
      v34 = v33 & (v53 + v34);
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( v4 == *v35 )
        goto LABEL_45;
      v53 = v55;
    }
    v6 = 0;
  }
  v7 = v5;
  if ( *(_BYTE *)v5 > 0x15u )
  {
LABEL_47:
    v37 = v33 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v38 = (__int64 **)(v32 + 16LL * v37);
    v39 = *v38;
    if ( v5 == *v38 )
    {
LABEL_48:
      v7 = v38[1];
    }
    else
    {
      v54 = 1;
      while ( v39 != (__int64 *)-4096LL )
      {
        v56 = v54 + 1;
        v37 = v33 & (v54 + v37);
        v38 = (__int64 **)(v32 + 16LL * v37);
        v39 = *v38;
        if ( v5 == *v38 )
          goto LABEL_48;
        v54 = v56;
      }
      v7 = 0;
    }
  }
LABEL_3:
  v8 = *a2;
  v9 = v8 - 29;
  if ( (unsigned __int8)v8 <= 0x1Cu )
    goto LABEL_19;
  switch ( *a2 )
  {
    case ')':
    case '+':
    case '-':
    case '/':
    case '2':
    case '5':
    case 'J':
    case 'K':
    case 'S':
      goto LABEL_27;
    case 'T':
    case 'U':
    case 'V':
      v10 = *((_QWORD *)a2 + 1);
      v11 = *(unsigned __int8 *)(v10 + 8);
      v12 = v11 - 17;
      v13 = *(_BYTE *)(v10 + 8);
      if ( (unsigned int)(v11 - 17) <= 1 )
        v13 = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
      if ( v13 <= 3u || v13 == 5 || (v13 & 0xFD) == 4 )
        goto LABEL_27;
      if ( (_BYTE)v11 == 15 )
      {
        if ( (*(_BYTE *)(v10 + 9) & 4) == 0 )
          goto LABEL_19;
        v82 = v7;
        v14 = sub_BCB420(*((_QWORD *)a2 + 1));
        v7 = v82;
        if ( !v14 )
        {
          v9 = *a2 - 29;
          goto LABEL_19;
        }
        v15 = *(__int64 **)(v10 + 16);
        v10 = *v15;
        v9 = *a2 - 29;
        v11 = *(unsigned __int8 *)(*v15 + 8);
        v12 = v11 - 17;
      }
      else if ( (_BYTE)v11 == 16 )
      {
        do
        {
          v10 = *(_QWORD *)(v10 + 24);
          LOBYTE(v11) = *(_BYTE *)(v10 + 8);
        }
        while ( (_BYTE)v11 == 16 );
        v12 = (unsigned __int8)v11 - 17;
      }
      if ( v12 <= 1 )
        LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
      if ( (unsigned __int8)v11 > 3u && (_BYTE)v11 != 5 && (v11 & 0xFD) != 4 )
      {
LABEL_19:
        v16 = *(_QWORD *)(a1 + 80);
        v85.m128i_i64[1] = 0;
        if ( !v7 )
          v7 = v5;
        v86 = 0;
        if ( !v6 )
          v6 = (__int64 *)v4;
        v85.m128i_i64[0] = v16;
        v87 = 0;
        v88 = 0;
        v89 = 0;
        v90 = 0;
        v91 = 0;
        v92 = 257;
        v17 = sub_101E7C0(v9, v6, v7, &v85);
        if ( v17 )
          goto LABEL_24;
LABEL_34:
        v23 = sub_30D1740(a1, v4);
        if ( v23 )
          sub_30D1890(a1, v23);
        v25 = sub_30D1740(a1, (__int64)v5);
        if ( v25 )
          sub_30D1890(a1, v25);
        v26 = *((_QWORD *)a2 + 1);
        v27 = *(_BYTE *)(v26 + 8);
        if ( v27 > 3u && v27 != 5 && (v27 & 0xFD) != 4 )
          return 0;
        v28 = sub_DFAF50(*(__int64 **)(a1 + 8), v26, v24);
        v30 = v29;
        if ( v29 || v28 != 4 )
          return 0;
        v41 = *a2;
        if ( (unsigned __int8)v41 <= 0x1Cu )
          goto LABEL_72;
        v42 = v41 - 29;
        switch ( *a2 )
        {
          case ')':
          case '+':
          case '-':
          case '/':
          case '2':
          case '5':
          case 'J':
          case 'K':
          case 'S':
            goto LABEL_61;
          case 'T':
          case 'U':
          case 'V':
            v49 = *((_QWORD *)a2 + 1);
            v50 = *(unsigned __int8 *)(v49 + 8);
            v51 = *(_BYTE *)(v49 + 8);
            if ( (unsigned int)(v50 - 17) <= 1 )
              v51 = *(_BYTE *)(**(_QWORD **)(v49 + 16) + 8LL);
            if ( v51 <= 3u || v51 == 5 || (v51 & 0xFD) == 4 )
              goto LABEL_62;
            if ( (_BYTE)v50 == 15 )
            {
              if ( (*(_BYTE *)(v49 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)a2 + 1)) )
                goto LABEL_72;
              v49 = **(_QWORD **)(v49 + 16);
            }
            else if ( (_BYTE)v50 == 16 )
            {
              do
                v49 = *(_QWORD *)(v49 + 24);
              while ( *(_BYTE *)(v49 + 8) == 16 );
            }
            if ( (unsigned int)*(unsigned __int8 *)(v49 + 8) - 17 <= 1 )
              v49 = **(_QWORD **)(v49 + 16);
            v52 = *(_BYTE *)(v49 + 8);
            if ( v52 > 3u && v52 != 5 && (v52 & 0xFD) != 4 )
              goto LABEL_72;
            v41 = *a2;
            if ( (unsigned __int8)v41 > 0x1Cu )
LABEL_61:
              v42 = v41 - 29;
            else
              v42 = *((unsigned __int16 *)a2 + 1);
LABEL_62:
            if ( v42 == 12 )
              return 0;
            if ( v42 != 16 )
              goto LABEL_72;
            v43 = a2[7] & 0x40;
            if ( (a2[1] & 0x10) != 0 )
            {
              if ( v43 )
                v44 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
              else
                v44 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
              v45 = *(unsigned __int8 **)v44;
              v46 = *v45;
              if ( (_BYTE)v46 == 18 )
              {
                if ( *((void **)v45 + 3) == sub_C33340() )
                  v47 = (unsigned __int8 *)*((_QWORD *)v45 + 4);
                else
                  v47 = v45 + 24;
                v48 = (v47[20] & 7) == 3;
                goto LABEL_71;
              }
              v67 = *((_QWORD *)v45 + 1);
              if ( (unsigned int)*(unsigned __int8 *)(v67 + 8) - 17 > 1 || (unsigned __int8)v46 > 0x15u )
                goto LABEL_72;
              v68 = (void **)sub_AD7630((__int64)v45, 0, v46);
              v69 = v68;
              if ( v68 && *(_BYTE *)v68 == 18 )
              {
                if ( v68[3] == sub_C33340() )
                  v70 = (void **)v69[4];
                else
                  v70 = v69 + 3;
                v48 = (*((_BYTE *)v70 + 20) & 7) == 3;
LABEL_71:
                if ( v48 )
                  return 0;
                goto LABEL_72;
              }
              if ( *(_BYTE *)(v67 + 8) != 17 )
                goto LABEL_72;
              v77 = *(_DWORD *)(v67 + 32);
              v72 = 0;
              v84 = v77;
              while ( v84 != v30 )
              {
                v78 = (void **)sub_AD69F0(v45, v30);
                v79 = v78;
                if ( !v78 )
                  goto LABEL_72;
                v80 = *(_BYTE *)v78;
                if ( v80 != 13 )
                {
                  if ( v80 != 18 )
                    goto LABEL_72;
                  v81 = v79[3] == sub_C33340() ? v79[4] : v79 + 3;
                  if ( (v81[20] & 7) != 3 )
                    goto LABEL_72;
                  v72 = 1;
                }
                ++v30;
              }
            }
            else
            {
              if ( v43 )
                v57 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
              else
                v57 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
              v58 = *(unsigned __int8 **)v57;
              v59 = *v58;
              if ( (_BYTE)v59 == 18 )
              {
                if ( *((void **)v58 + 3) == sub_C33340() )
                {
                  v61 = (unsigned __int8 *)*((_QWORD *)v58 + 4);
                  if ( (v61[20] & 7) != 3 )
                    goto LABEL_72;
                }
                else
                {
                  v60 = v58[44];
                  v61 = v58 + 24;
                  if ( (v60 & 7) != 3 )
                    goto LABEL_72;
                }
                if ( (v61[20] & 8) == 0 )
                  goto LABEL_72;
                return 0;
              }
              v62 = *((_QWORD *)v58 + 1);
              if ( (unsigned int)*(unsigned __int8 *)(v62 + 8) - 17 > 1 || (unsigned __int8)v59 > 0x15u )
                goto LABEL_72;
              v63 = (void **)sub_AD7630((__int64)v58, 0, v59);
              v64 = v63;
              if ( v63 && *(_BYTE *)v63 == 18 )
              {
                if ( v63[3] == sub_C33340() )
                {
                  v66 = (void **)v64[4];
                  if ( (*((_BYTE *)v66 + 20) & 7) != 3 )
                    goto LABEL_72;
                }
                else
                {
                  v65 = *((_BYTE *)v64 + 44);
                  v66 = v64 + 3;
                  if ( (v65 & 7) != 3 )
                    goto LABEL_72;
                }
                if ( (*((_BYTE *)v66 + 20) & 8) != 0 )
                  return 0;
                goto LABEL_72;
              }
              if ( *(_BYTE *)(v62 + 8) != 17 )
                goto LABEL_72;
              v71 = *(_DWORD *)(v62 + 32);
              v72 = 0;
              v83 = v71;
              while ( v83 != v30 )
              {
                v73 = (_BYTE *)sub_AD69F0(v58, v30);
                v74 = v73;
                if ( !v73 )
                  goto LABEL_72;
                v75 = *v73;
                if ( v75 != 13 )
                {
                  if ( v75 != 18 )
                    goto LABEL_72;
                  if ( *((void **)v74 + 3) == sub_C33340() )
                  {
                    v76 = (_BYTE *)*((_QWORD *)v74 + 4);
                    if ( (v76[20] & 7) != 3 )
                      goto LABEL_72;
                  }
                  else
                  {
                    if ( (v74[44] & 7) != 3 )
                      goto LABEL_72;
                    v76 = v74 + 24;
                  }
                  if ( (v76[20] & 8) == 0 )
                    goto LABEL_72;
                  v72 = 1;
                }
                ++v30;
              }
            }
            if ( v72 )
              return 0;
LABEL_72:
            (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 96LL))(a1);
            return 0;
          default:
            goto LABEL_72;
        }
      }
LABEL_27:
      v20 = *(_QWORD *)(a1 + 80);
      v92 = 257;
      v21 = a2[1];
      v85 = (__m128i)v20;
      v86 = 0;
      v22 = v21 >> 1;
      v87 = 0;
      v88 = 0;
      if ( v22 == 127 )
        LOBYTE(v22) = -1;
      v89 = 0;
      if ( !v7 )
        v7 = v5;
      v90 = 0;
      if ( !v6 )
        v6 = (__int64 *)v4;
      v91 = 0;
      v17 = sub_101E830(v9, v6, v7, v22, &v85);
      if ( !v17 )
        goto LABEL_34;
LABEL_24:
      v18 = 1;
      if ( *v17 <= 0x15u )
      {
        v85.m128i_i64[0] = (__int64)a2;
        *sub_30D9190(a1 + 136, v85.m128i_i64) = (__int64)v17;
      }
      return v18;
    default:
      goto LABEL_19;
  }
}
