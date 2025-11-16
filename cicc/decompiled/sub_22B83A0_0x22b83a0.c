// Function: sub_22B83A0
// Address: 0x22b83a0
//
__int64 __fastcall sub_22B83A0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 result; // rax
  int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // r14
  unsigned __int8 *v10; // r12
  __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rax
  unsigned __int8 *v15; // r8
  int v16; // edi
  __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned int v19; // edx
  __int64 v20; // rax
  _BYTE *v21; // r10
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int8 v28; // al
  __int64 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // r8
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 *v44; // r9
  __int64 v45; // rcx
  __int64 *v46; // r14
  int *v47; // r13
  __int64 *v48; // r12
  __int64 v49; // r15
  int *v50; // rbx
  int v51; // eax
  __int64 v52; // r10
  int v53; // r11d
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  int v57; // r9d
  int v58; // r9d
  __int64 *v59; // [rsp+0h] [rbp-170h]
  __int64 *v60; // [rsp+8h] [rbp-168h]
  int *v61; // [rsp+10h] [rbp-160h]
  __int64 v62; // [rsp+18h] [rbp-158h]
  __int64 v63; // [rsp+20h] [rbp-150h]
  int *v64; // [rsp+28h] [rbp-148h]
  unsigned int v65; // [rsp+30h] [rbp-140h]
  int v66; // [rsp+34h] [rbp-13Ch]
  _BYTE *v69; // [rsp+48h] [rbp-128h]
  __int64 v70; // [rsp+48h] [rbp-128h]
  __int64 *v71; // [rsp+48h] [rbp-128h]
  int v72; // [rsp+58h] [rbp-118h] BYREF
  int v73; // [rsp+5Ch] [rbp-114h] BYREF
  _QWORD v74[2]; // [rsp+60h] [rbp-110h] BYREF
  __int64 *v75[19]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v76; // [rsp+108h] [rbp-68h]
  __int64 v77; // [rsp+110h] [rbp-60h]
  __int64 v78; // [rsp+120h] [rbp-50h]
  __int64 v79; // [rsp+128h] [rbp-48h]
  __int64 v80; // [rsp+130h] [rbp-40h]

  result = 0;
  v5 = *(_DWORD *)(a1 + 4);
  if ( v5 == *(_DWORD *)(a2 + 4) )
  {
    v6 = a1;
    v7 = a2;
    if ( *(_DWORD *)(a1 + 40) == *(_DWORD *)(a2 + 40) )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = *(_QWORD *)(a2 + 8);
      v66 = *(_DWORD *)a1;
      v65 = *(_DWORD *)a1 + v5;
      if ( v65 <= *(_DWORD *)a1 )
        return 1;
      while ( 1 )
      {
        if ( !(unsigned __int8)sub_22AF4E0(v8, v9) )
          return 0;
        v10 = *(unsigned __int8 **)(v8 + 16);
        v69 = *(_BYTE **)(v9 + 16);
        if ( !*(_BYTE *)(v8 + 72) || !*(_BYTE *)(v9 + 72) )
          return 0;
        v11 = *(unsigned int *)(v6 + 48);
        v12 = *(_QWORD *)(v6 + 32);
        v74[0] = *(_QWORD *)(v8 + 24);
        v74[1] = *(unsigned int *)(v8 + 32);
        v75[0] = *(__int64 **)(v9 + 24);
        v75[1] = (__int64 *)*(unsigned int *)(v9 + 32);
        if ( (_DWORD)v11 )
        {
          v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v14 = v12 + 16LL * v13;
          v15 = *(unsigned __int8 **)v14;
          if ( v10 == *(unsigned __int8 **)v14 )
            goto LABEL_9;
          v56 = 1;
          while ( v15 != (unsigned __int8 *)-4096LL )
          {
            v57 = v56 + 1;
            v13 = (v11 - 1) & (v56 + v13);
            v14 = v12 + 16LL * v13;
            v15 = *(unsigned __int8 **)v14;
            if ( v10 == *(unsigned __int8 **)v14 )
              goto LABEL_9;
            v56 = v57;
          }
        }
        v14 = v12 + 16 * v11;
LABEL_9:
        v16 = *(_DWORD *)(v14 + 8);
        v17 = *(unsigned int *)(v7 + 48);
        v18 = *(_QWORD *)(v7 + 32);
        v72 = v16;
        if ( !(_DWORD)v17 )
          goto LABEL_57;
        v19 = (v17 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v20 = v18 + 16LL * v19;
        v21 = *(_BYTE **)v20;
        if ( v69 != *(_BYTE **)v20 )
        {
          v55 = 1;
          while ( v21 != (_BYTE *)-4096LL )
          {
            v58 = v55 + 1;
            v19 = (v17 - 1) & (v55 + v19);
            v20 = v18 + 16LL * v19;
            v21 = *(_BYTE **)v20;
            if ( v69 == *(_BYTE **)v20 )
              goto LABEL_11;
            v55 = v58;
          }
LABEL_57:
          v20 = v18 + 16 * v17;
        }
LABEL_11:
        v73 = *(_DWORD *)(v20 + 8);
        if ( !(unsigned __int8)sub_22B7A50(v16, &v73, (__int64)a3) )
          return 0;
        v22 = (__int64)&v72;
        if ( !(unsigned __int8)sub_22B7A50(v73, &v72, (__int64)a4) )
          return 0;
        v23 = (__int64)v10;
        if ( sub_B46D50(v10) )
        {
          v24 = *v10;
          if ( (unsigned __int8)v24 <= 0x1Cu )
          {
LABEL_32:
            v75[2] = (__int64 *)v7;
            v75[3] = (__int64 *)v75;
            v75[6] = (__int64 *)v6;
            v75[4] = a4;
            v75[7] = v74;
            v75[8] = a3;
            if ( !(unsigned __int8)sub_22B7530(
                                     v23,
                                     v22,
                                     v24,
                                     v25,
                                     v26,
                                     v27,
                                     v6,
                                     (__int64)v74,
                                     (__int64)a3,
                                     v7,
                                     v75,
                                     (__int64)a4) )
              return 0;
            goto LABEL_53;
          }
          switch ( (char)v24 )
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
              break;
            case 'T':
            case 'U':
            case 'V':
              v23 = *((_QWORD *)v10 + 1);
              v25 = *(unsigned __int8 *)(v23 + 8);
              v22 = (unsigned int)(v25 - 17);
              v28 = *(_BYTE *)(v23 + 8);
              if ( (unsigned int)v22 <= 1 )
                v28 = *(_BYTE *)(**(_QWORD **)(v23 + 16) + 8LL);
              if ( v28 <= 3u || v28 == 5 || (v28 & 0xFD) == 4 )
                break;
              if ( (_BYTE)v25 != 15 )
              {
                if ( (_BYTE)v25 == 16 )
                {
                  do
                  {
                    v23 = *(_QWORD *)(v23 + 24);
                    v25 = *(unsigned __int8 *)(v23 + 8);
                  }
                  while ( (_BYTE)v25 == 16 );
                  v22 = (unsigned int)(unsigned __int8)v25 - 17;
                }
LABEL_25:
                if ( (unsigned int)v22 <= 1 )
                  v25 = *(unsigned __int8 *)(**(_QWORD **)(v23 + 16) + 8LL);
                if ( (unsigned __int8)v25 <= 3u )
                  break;
                if ( (_BYTE)v25 == 5 )
                  break;
                v25 = (unsigned int)v25 & 0xFFFFFFFD;
                if ( (_BYTE)v25 == 4 )
                  break;
                goto LABEL_30;
              }
              if ( (*(_BYTE *)(v23 + 9) & 4) == 0 )
                goto LABEL_31;
              if ( sub_BCB420(v23) )
              {
                v29 = *(__int64 **)(v23 + 16);
                v23 = *v29;
                v25 = *(unsigned __int8 *)(*v29 + 8);
                v22 = (unsigned int)(v25 - 17);
                goto LABEL_25;
              }
LABEL_30:
              v24 = *v10;
LABEL_31:
              if ( (_BYTE)v24 != 85 )
                goto LABEL_32;
              v30 = *((_QWORD *)v10 - 4);
              if ( !v30 )
                goto LABEL_32;
              if ( *(_BYTE *)v30 )
                goto LABEL_32;
              v22 = *((_QWORD *)v10 + 10);
              if ( *(_QWORD *)(v30 + 24) != v22 || (*(_BYTE *)(v30 + 33) & 0x20) == 0 )
                goto LABEL_32;
              break;
            default:
              goto LABEL_31;
          }
        }
        v75[10] = (__int64 *)v7;
        v75[11] = (__int64 *)v75;
        v75[14] = (__int64 *)v6;
        v75[12] = a4;
        v75[15] = v74;
        v75[16] = a3;
        if ( !(unsigned __int8)sub_22B7360(
                                 v23,
                                 v22,
                                 v24,
                                 v25,
                                 v26,
                                 v27,
                                 v6,
                                 (__int64)v74,
                                 (__int64)a3,
                                 v7,
                                 v75,
                                 (__int64)a4) )
          return 0;
        if ( *v10 == 31 )
        {
          if ( *v69 != 31 )
            goto LABEL_53;
        }
        else if ( *v10 != 84 || *v69 != 84 )
        {
          goto LABEL_53;
        }
        v31 = sub_22AE7B0(v8);
        v70 = v32;
        v33 = (__int64 *)v31;
        v34 = sub_22AE7B0(v9);
        v36 = *(unsigned int *)(v8 + 136);
        v37 = v70;
        v38 = v34;
        v39 = *(unsigned int *)(v9 + 136);
        v40 = v35;
        if ( (_DWORD)v39 != (_DWORD)v36 && v35 != v70 )
          return 0;
        v41 = *(_QWORD *)(v8 + 128);
        v64 = (int *)(v41 + 4 * v36);
        if ( (int *)v41 != v64 )
        {
          v42 = *(_QWORD *)(v9 + 128);
          v43 = v38 + 8 * v40;
          v63 = v9;
          v44 = (__int64 *)v7;
          v59 = (__int64 *)v43;
          v45 = (__int64)&v33[v70];
          v46 = v33;
          v47 = *(int **)(v8 + 128);
          v60 = (__int64 *)v45;
          v48 = (__int64 *)v38;
          v61 = (int *)(v42 + 4 * v39);
          v62 = v8;
          v49 = v6;
          v50 = (int *)v42;
          do
          {
            if ( v50 == v61 || v46 == v60 || v48 == v59 )
              break;
            v51 = *v47;
            v52 = *v48;
            v75[18] = v44;
            v53 = *v50;
            v78 = v49;
            LODWORD(v79) = v51;
            v54 = *v46;
            LODWORD(v76) = v53;
            v77 = v52;
            v80 = v54;
            v71 = v44;
            if ( !(unsigned __int8)sub_22B4CE0(
                                     v38,
                                     v42,
                                     v41,
                                     v45,
                                     v37,
                                     (__int64)v44,
                                     v49,
                                     v79,
                                     v54,
                                     (__int64)v44,
                                     v53,
                                     v52) )
              return 0;
            ++v46;
            ++v48;
            ++v50;
            ++v47;
            v44 = v71;
          }
          while ( v64 != v47 );
          v6 = v49;
          v9 = v63;
          v8 = v62;
          v7 = (__int64)v44;
        }
LABEL_53:
        ++v66;
        v8 = *(_QWORD *)(v8 + 8);
        v9 = *(_QWORD *)(v9 + 8);
        if ( v65 == v66 )
          return 1;
      }
    }
  }
  return result;
}
