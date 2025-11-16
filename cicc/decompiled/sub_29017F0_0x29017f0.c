// Function: sub_29017F0
// Address: 0x29017f0
//
void __fastcall sub_29017F0(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rdx
  _QWORD *v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // rcx
  __int64 *v19; // r12
  __int64 v20; // rbx
  unsigned int v21; // r8d
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rbx
  __int64 v27; // r10
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rsi
  int v32; // edi
  __int64 v33; // rax
  unsigned int v34; // edi
  _QWORD *v35; // rax
  __int64 v36; // r14
  __int64 v37; // rbx
  int v38; // edx
  unsigned int v39; // ecx
  unsigned __int8 v40; // al
  __int64 *v41; // rax
  __int64 v42; // rdx
  int v43; // r12d
  __int64 v44; // r12
  __int64 v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // r9
  int v49; // ebx
  __int64 *v50; // r10
  int v51; // eax
  __int64 *v52; // rax
  unsigned int v53; // r11d
  int v54; // r10d
  __int64 v55; // rax
  __int64 *v56; // rdi
  __int64 v57; // rbx
  bool v58; // zf
  __int64 *v59; // rax
  int v60; // edx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // [rsp+0h] [rbp-1A0h]
  __int64 v64; // [rsp+8h] [rbp-198h]
  __int64 **v65; // [rsp+10h] [rbp-190h]
  __int64 v66; // [rsp+20h] [rbp-180h]
  int v70; // [rsp+50h] [rbp-150h]
  __int64 v71; // [rsp+58h] [rbp-148h]
  __int64 v72; // [rsp+60h] [rbp-140h]
  __int64 v73; // [rsp+78h] [rbp-128h]
  __int64 v74; // [rsp+78h] [rbp-128h]
  unsigned int v76; // [rsp+88h] [rbp-118h]
  unsigned int v77; // [rsp+8Ch] [rbp-114h]
  __int64 v78; // [rsp+A8h] [rbp-F8h] BYREF
  __int64 v79; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v80; // [rsp+B8h] [rbp-E8h]
  __int64 v81; // [rsp+C0h] [rbp-E0h]
  __int64 v82; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v83; // [rsp+D8h] [rbp-C8h]
  __int64 v84; // [rsp+E0h] [rbp-C0h]
  unsigned int v85; // [rsp+E8h] [rbp-B8h]
  unsigned __int64 v86[2]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+100h] [rbp-A0h] BYREF
  _QWORD v88[4]; // [rsp+110h] [rbp-90h] BYREF
  __int16 v89; // [rsp+130h] [rbp-70h]
  _QWORD v90[4]; // [rsp+140h] [rbp-60h] BYREF
  __int16 v91; // [rsp+160h] [rbp-40h]

  if ( a2 )
  {
    v6 = 0;
    v82 = 0;
    v65 = (__int64 **)sub_B43CA0(a4);
    v83 = 0;
    v7 = (8 * a2) & 0xFFFFFFFFFFFFFFE0LL;
    v84 = 0;
    v64 = (__int64)(8 * a2) >> 3;
    v8 = (_QWORD *)((char *)a1 + v7);
    v85 = 0;
    v66 = (__int64)(8 * a2) >> 5;
    v77 = 0;
    v63 = (__int64)(8 * a2 - v7) >> 3;
    while ( 1 )
    {
      v9 = v6;
      v10 = *(_QWORD *)(a3 + 8 * v6);
      if ( v66 > 0 )
      {
        v11 = a1;
        while ( v10 != *v11 )
        {
          if ( v10 == v11[1] )
          {
            v12 = v11 + 1 - a1;
            goto LABEL_11;
          }
          if ( v10 == v11[2] )
          {
            v12 = v11 + 2 - a1;
            goto LABEL_11;
          }
          if ( v10 == v11[3] )
          {
            v12 = v11 + 3 - a1;
            goto LABEL_11;
          }
          v11 += 4;
          if ( v8 == v11 )
          {
            v62 = v63;
            goto LABEL_81;
          }
        }
LABEL_10:
        LODWORD(v12) = v11 - a1;
        goto LABEL_11;
      }
      v62 = v64;
      v11 = a1;
LABEL_81:
      if ( v62 == 2 )
        goto LABEL_87;
      if ( v62 == 3 )
        break;
      if ( v62 != 1 )
      {
        LODWORD(v12) = v64;
        goto LABEL_11;
      }
LABEL_89:
      LODWORD(v12) = v64;
      if ( v10 == *v11 )
        goto LABEL_10;
LABEL_11:
      v13 = sub_BCB2D0((_QWORD *)a5[9]);
      v14 = sub_ACD640(v13, (unsigned int)v12, 0);
      v15 = sub_BCB2D0((_QWORD *)a5[9]);
      v16 = sub_ACD640(v15, v6, 0);
      v17 = v85;
      v18 = v83;
      v73 = v16;
      v19 = &a1[v9];
      v20 = *(_QWORD *)(a1[v9] + 8LL);
      v78 = v20;
      if ( v85 )
      {
        v21 = v85 - 1;
        v22 = (v85 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v23 = (__int64 *)(v83 + 16LL * v22);
        v24 = *v23;
        if ( v20 == *v23 )
          goto LABEL_13;
        v48 = *v23;
        v53 = (v85 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v54 = 1;
        while ( v48 != -4096 )
        {
          v53 = v21 & (v54 + v53);
          v48 = *(_QWORD *)(v83 + 16LL * v53);
          if ( v20 == v48 )
            goto LABEL_49;
          ++v54;
        }
      }
      v55 = v20;
      if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
      {
        v55 = **(_QWORD **)(v20 + 16);
        if ( (unsigned int)*(unsigned __int8 *)(v55 + 8) - 17 <= 1 )
          v55 = **(_QWORD **)(v55 + 16);
      }
      v56 = (__int64 *)sub_BCE3C0(*v65, *(_DWORD *)(v55 + 8) >> 8);
      if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
        v56 = (__int64 *)sub_BCDA70(v56, *(_DWORD *)(v20 + 32));
      v90[0] = v56;
      v57 = sub_B6E160((__int64 *)v65, 0x95u, (__int64)v90, 1);
      v58 = (unsigned __int8)sub_2901270((__int64)&v82, &v78, v88) == 0;
      v59 = (__int64 *)v88[0];
      if ( v58 )
      {
        v90[0] = v88[0];
        ++v82;
        v60 = v84 + 1;
        if ( 4 * ((int)v84 + 1) >= 3 * v85 )
        {
          sub_2901610((__int64)&v82, 2 * v85);
          sub_2901270((__int64)&v82, &v78, v90);
          v60 = v84 + 1;
          v59 = (__int64 *)v90[0];
        }
        else if ( v85 - HIDWORD(v84) - v60 <= v85 >> 3 )
        {
          sub_2901610((__int64)&v82, v85);
          sub_2901270((__int64)&v82, &v78, v90);
          v60 = v84 + 1;
          v59 = (__int64 *)v90[0];
        }
        LODWORD(v84) = v60;
        if ( *v59 != -4096 )
          --HIDWORD(v84);
        v61 = v78;
        v59[1] = 0;
        *v59 = v61;
      }
      v59[1] = v57;
      v17 = v85;
      if ( !v85 )
      {
        ++v82;
        v90[0] = 0;
        goto LABEL_75;
      }
      v48 = v78;
      v21 = v85 - 1;
      v18 = v83;
      v22 = (v85 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
      v23 = (__int64 *)(v83 + 16LL * v22);
      v24 = *v23;
      if ( v78 != *v23 )
      {
LABEL_49:
        v49 = 1;
        v50 = 0;
        while ( v24 != -4096 )
        {
          if ( v24 == -8192 && !v50 )
            v50 = v23;
          v22 = v21 & (v49 + v22);
          v23 = (__int64 *)(v18 + 16LL * v22);
          v24 = *v23;
          if ( *v23 == v48 )
            goto LABEL_13;
          ++v49;
        }
        if ( !v50 )
          v50 = v23;
        ++v82;
        v51 = v84 + 1;
        v90[0] = v50;
        if ( 4 * ((int)v84 + 1) >= 3 * v17 )
        {
LABEL_75:
          sub_2901610((__int64)&v82, 2 * v17);
        }
        else
        {
          if ( v17 - (v51 + HIDWORD(v84)) > v17 >> 3 )
          {
LABEL_55:
            LODWORD(v84) = v51;
            if ( *v50 != -4096 )
              --HIDWORD(v84);
            *v50 = v48;
            v50[1] = 0;
            sub_28FF1A0((__int64)v86, *v19, ".relocated", (void *)0xA, byte_3F871B3, 0);
            v80 = v14;
            v89 = 260;
            v79 = a4;
            v88[0] = v86;
            v26 = 0;
            v81 = v73;
LABEL_58:
            v27 = 0;
            goto LABEL_15;
          }
          sub_2901610((__int64)&v82, v17);
        }
        sub_2901270((__int64)&v82, &v78, v90);
        v48 = v78;
        v50 = (__int64 *)v90[0];
        v51 = v84 + 1;
        goto LABEL_55;
      }
LABEL_13:
      v25 = *v19;
      v26 = v23[1];
      v19 = (__int64 *)v86;
      sub_28FF1A0((__int64)v86, v25, ".relocated", (void *)0xA, byte_3F871B3, 0);
      v88[0] = v86;
      v89 = 260;
      v79 = a4;
      v80 = v14;
      v81 = v73;
      if ( !v26 )
        goto LABEL_58;
      v27 = *(_QWORD *)(v26 + 24);
LABEL_15:
      v28 = a5[15];
      v91 = 257;
      v29 = a5[14];
      v30 = v29 + 56 * v28;
      if ( v29 == v30 )
      {
        v70 = 4;
        v34 = 4;
      }
      else
      {
        v31 = a5[14];
        v32 = 0;
        do
        {
          v33 = *(_QWORD *)(v31 + 40) - *(_QWORD *)(v31 + 32);
          v31 += 56;
          v32 += v33 >> 3;
        }
        while ( v30 != v31 );
        v34 = v32 + 4;
        v70 = v34 & 0x7FFFFFF;
      }
      v71 = a5[14];
      v72 = v27;
      LOBYTE(v19) = 16 * (_DWORD)v28 != 0;
      v74 = v28;
      v35 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v28) << 32) | v34);
      v36 = (__int64)v35;
      if ( v35 )
      {
        v76 = v76 & 0xE0000000 | ((_DWORD)v19 << 28) | v70;
        sub_B44260((__int64)v35, **(_QWORD **)(v72 + 16), 56, v76, 0, 0);
        *(_QWORD *)(v36 + 72) = 0;
        sub_B4A290(v36, v72, v26, &v79, 3, (__int64)v90, v71, v74);
      }
      if ( *((_BYTE *)a5 + 108) )
      {
        v52 = (__int64 *)sub_BD5C60(v36);
        *(_QWORD *)(v36 + 72) = sub_A7A090((__int64 *)(v36 + 72), v52, -1, 72);
      }
      if ( *(_BYTE *)v36 > 0x1Cu )
      {
        switch ( *(_BYTE *)v36 )
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
            goto LABEL_39;
          case 'T':
          case 'U':
          case 'V':
            v37 = *(_QWORD *)(v36 + 8);
            v38 = *(unsigned __int8 *)(v37 + 8);
            v39 = v38 - 17;
            v40 = *(_BYTE *)(v37 + 8);
            if ( (unsigned int)(v38 - 17) <= 1 )
              v40 = *(_BYTE *)(**(_QWORD **)(v37 + 16) + 8LL);
            if ( v40 <= 3u || v40 == 5 || (v40 & 0xFD) == 4 )
              goto LABEL_39;
            if ( (_BYTE)v38 == 15 )
            {
              if ( (*(_BYTE *)(v37 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v36 + 8)) )
                break;
              v41 = *(__int64 **)(v37 + 16);
              v37 = *v41;
              v38 = *(unsigned __int8 *)(*v41 + 8);
              v39 = v38 - 17;
            }
            else if ( (_BYTE)v38 == 16 )
            {
              do
              {
                v37 = *(_QWORD *)(v37 + 24);
                LOBYTE(v38) = *(_BYTE *)(v37 + 8);
              }
              while ( (_BYTE)v38 == 16 );
              v39 = (unsigned __int8)v38 - 17;
            }
            if ( v39 <= 1 )
              LOBYTE(v38) = *(_BYTE *)(**(_QWORD **)(v37 + 16) + 8LL);
            if ( (unsigned __int8)v38 <= 3u || (_BYTE)v38 == 5 || (v38 & 0xFD) == 4 )
            {
LABEL_39:
              v42 = a5[12];
              v43 = *((_DWORD *)a5 + 26);
              if ( v42 )
                sub_B99FD0(v36, 3u, v42);
              sub_B45150(v36, v43);
            }
            break;
          default:
            break;
        }
      }
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a5[11] + 16LL))(
        a5[11],
        v36,
        v88,
        a5[7],
        a5[8]);
      v44 = *a5;
      v45 = *a5 + 16LL * *((unsigned int *)a5 + 2);
      if ( *a5 != v45 )
      {
        do
        {
          v46 = *(_QWORD *)(v44 + 8);
          v47 = *(_DWORD *)v44;
          v44 += 16;
          sub_B99FD0(v36, v47, v46);
        }
        while ( v45 != v44 );
      }
      if ( (__int64 *)v86[0] != &v87 )
        j_j___libc_free_0(v86[0]);
      v6 = ++v77;
      *(_WORD *)(v36 + 2) = *(_WORD *)(v36 + 2) & 0xF003 | 0x24;
      if ( a2 <= v77 )
      {
        sub_C7D6A0(v83, 16LL * v85, 8);
        return;
      }
    }
    if ( v10 == *v11 )
      goto LABEL_10;
    ++v11;
LABEL_87:
    if ( v10 == *v11 )
      goto LABEL_10;
    ++v11;
    goto LABEL_89;
  }
}
