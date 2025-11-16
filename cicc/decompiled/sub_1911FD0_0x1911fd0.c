// Function: sub_1911FD0
// Address: 0x1911fd0
//
__int64 __fastcall sub_1911FD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // r8d
  unsigned int v7; // ecx
  unsigned int *v8; // rax
  __int64 v9; // r9
  unsigned int v10; // r13d
  int v12; // eax
  int v13; // eax
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // r8d
  int v20; // r9d
  unsigned int v21; // eax
  unsigned int v22; // esi
  unsigned int v23; // r15d
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // rdi
  unsigned int v29; // ecx
  __int64 *v30; // rax
  __int64 v31; // r9
  unsigned int v32; // ecx
  __int64 v33; // rax
  __int64 v34; // r13
  int v35; // edx
  unsigned int v36; // esi
  __int64 v37; // r8
  unsigned int v38; // ecx
  int *v39; // rax
  int v40; // edi
  int v41; // r10d
  int v42; // r11d
  __int64 *v43; // r10
  int v44; // ebx
  int v45; // edi
  int v46; // r11d
  __int64 *v47; // r10
  int v48; // edx
  int v49; // edi
  int v50; // r11d
  __int64 *v51; // r10
  int v52; // edi
  int v53; // edi
  int v54; // r11d
  int *v55; // r10
  int v56; // edx
  int v57; // ecx
  int v58; // edx
  int v59; // r10d
  __int64 v60; // r9
  int v61; // edi
  int v62; // edi
  int v63; // eax
  __int64 v64; // [rsp+8h] [rbp-B8h] BYREF
  int v65; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+18h] [rbp-A8h]
  char v67; // [rsp+20h] [rbp-A0h]
  _BYTE *v68; // [rsp+28h] [rbp-98h] BYREF
  __int64 v69; // [rsp+30h] [rbp-90h]
  _BYTE v70[24]; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v71[2]; // [rsp+50h] [rbp-70h] BYREF
  char v72; // [rsp+60h] [rbp-60h]
  char *v73; // [rsp+68h] [rbp-58h] BYREF
  char v74; // [rsp+78h] [rbp-48h] BYREF

  v2 = a2;
  v64 = a2;
  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v8 = (unsigned int *)(v5 + 16LL * v7);
    v9 = *(_QWORD *)v8;
    if ( v2 == *(_QWORD *)v8 )
    {
LABEL_3:
      if ( v8 != (unsigned int *)(v5 + 16LL * v4) )
        return v8[2];
      if ( *(_BYTE *)(v2 + 16) > 0x17u )
      {
        v67 = 0;
        v68 = v70;
        v69 = 0x400000000LL;
        v63 = *(unsigned __int8 *)(v2 + 16);
        v65 = -3;
        switch ( v63 )
        {
          case '#':
          case '$':
          case '%':
          case '&':
          case '\'':
          case '(':
          case ')':
          case '*':
          case '+':
          case ',':
          case '-':
          case '.':
          case '/':
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '8':
          case '<':
          case '=':
          case '>':
          case '?':
          case '@':
          case 'A':
          case 'B':
          case 'C':
          case 'D':
          case 'E':
          case 'F':
          case 'G':
          case 'H':
          case 'K':
          case 'L':
          case 'O':
          case 'S':
          case 'T':
          case 'U':
          case 'W':
            goto LABEL_13;
          case 'M':
            goto LABEL_27;
          case 'N':
            goto LABEL_26;
          case 'V':
            goto LABEL_25;
          default:
            goto LABEL_23;
        }
      }
      goto LABEL_11;
    }
    v13 = 1;
    while ( v9 != -8 )
    {
      v41 = v13 + 1;
      v7 = v6 & (v13 + v7);
      v8 = (unsigned int *)(v5 + 16LL * v7);
      v9 = *(_QWORD *)v8;
      if ( v2 == *(_QWORD *)v8 )
        goto LABEL_3;
      v13 = v41;
    }
    if ( *(_BYTE *)(v2 + 16) <= 0x17u )
    {
LABEL_11:
      v14 = v6 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v15 = (__int64 *)(v5 + 16LL * v14);
      v16 = *v15;
      if ( v2 == *v15 )
      {
LABEL_12:
        v10 = *(_DWORD *)(a1 + 208);
        *((_DWORD *)v15 + 2) = v10;
        *(_DWORD *)(a1 + 208) = v10 + 1;
        return v10;
      }
      v42 = 1;
      v43 = 0;
      while ( v16 != -8 )
      {
        if ( v16 == -16 && !v43 )
          v43 = v15;
        v14 = v6 & (v42 + v14);
        v15 = (__int64 *)(v5 + 16LL * v14);
        v16 = *v15;
        if ( v2 == *v15 )
          goto LABEL_12;
        ++v42;
      }
      v44 = *(_DWORD *)(a1 + 16);
      if ( v43 )
        v15 = v43;
      ++*(_QWORD *)a1;
      v45 = v44 + 1;
      if ( 4 * (v44 + 1) < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(a1 + 20) - v45 > v4 >> 3 )
        {
LABEL_40:
          *(_DWORD *)(a1 + 16) = v45;
          if ( *v15 != -8 )
            --*(_DWORD *)(a1 + 20);
          *v15 = v2;
          *((_DWORD *)v15 + 2) = 0;
          goto LABEL_12;
        }
LABEL_45:
        sub_177C7D0(a1, v4);
        sub_190E590(a1, &v64, v71);
        v15 = (__int64 *)v71[0];
        v2 = v64;
        v45 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_40;
      }
LABEL_44:
      v4 *= 2;
      goto LABEL_45;
    }
  }
  else if ( *(_BYTE *)(v2 + 16) <= 0x17u )
  {
    ++*(_QWORD *)a1;
    goto LABEL_44;
  }
  v67 = 0;
  v68 = v70;
  v69 = 0x400000000LL;
  v12 = *(unsigned __int8 *)(v2 + 16);
  v65 = -3;
  switch ( v12 )
  {
    case '#':
    case '$':
    case '%':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '8':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'K':
    case 'L':
    case 'O':
    case 'S':
    case 'T':
    case 'U':
    case 'W':
LABEL_13:
      sub_19127E0(v71, a1);
      goto LABEL_14;
    case 'M':
LABEL_27:
      if ( !v4 )
      {
        ++*(_QWORD *)a1;
        goto LABEL_92;
      }
      v32 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v33 = v5 + 16LL * v32;
      v34 = *(_QWORD *)v33;
      if ( v2 == *(_QWORD *)v33 )
        goto LABEL_29;
      v59 = 1;
      v60 = 0;
      while ( 1 )
      {
        if ( v34 == -8 )
        {
          v61 = *(_DWORD *)(a1 + 16);
          if ( v60 )
            v33 = v60;
          ++*(_QWORD *)a1;
          v62 = v61 + 1;
          if ( 4 * v62 < 3 * v4 )
          {
            if ( v4 - *(_DWORD *)(a1 + 20) - v62 > v4 >> 3 )
            {
LABEL_88:
              *(_DWORD *)(a1 + 16) = v62;
              if ( *(_QWORD *)v33 != -8 )
                --*(_DWORD *)(a1 + 20);
              *(_QWORD *)v33 = v2;
              v34 = v64;
              *(_DWORD *)(v33 + 8) = 0;
              break;
            }
LABEL_93:
            sub_177C7D0(a1, v4);
            sub_190E590(a1, &v64, v71);
            v33 = v71[0];
            v2 = v64;
            v62 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_88;
          }
LABEL_92:
          v4 *= 2;
          goto LABEL_93;
        }
        if ( v34 == -16 && !v60 )
          v60 = v33;
        v32 = (v4 - 1) & (v59 + v32);
        v33 = v5 + 16LL * v32;
        v34 = *(_QWORD *)v33;
        if ( v2 == *(_QWORD *)v33 )
          break;
        ++v59;
      }
LABEL_29:
      v35 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)(v33 + 8) = v35;
      v36 = *(_DWORD *)(a1 + 144);
      if ( !v36 )
      {
        ++*(_QWORD *)(a1 + 120);
LABEL_80:
        v36 *= 2;
        goto LABEL_81;
      }
      v37 = *(_QWORD *)(a1 + 128);
      v38 = (v36 - 1) & (37 * v35);
      v39 = (int *)(v37 + 16LL * v38);
      v40 = *v39;
      if ( v35 == *v39 )
        goto LABEL_31;
      v54 = 1;
      v55 = 0;
      while ( v40 != -1 )
      {
        if ( !v55 && v40 == -2 )
          v55 = v39;
        v38 = (v36 - 1) & (v54 + v38);
        v39 = (int *)(v37 + 16LL * v38);
        v40 = *v39;
        if ( v35 == *v39 )
          goto LABEL_31;
        ++v54;
      }
      v56 = *(_DWORD *)(a1 + 136);
      if ( v55 )
        v39 = v55;
      ++*(_QWORD *)(a1 + 120);
      v57 = v56 + 1;
      if ( 4 * (v56 + 1) >= 3 * v36 )
        goto LABEL_80;
      if ( v36 - *(_DWORD *)(a1 + 140) - v57 <= v36 >> 3 )
      {
LABEL_81:
        sub_1910EE0(a1 + 120, v36);
        sub_190E640(a1 + 120, (int *)(a1 + 208), v71);
        v39 = (int *)v71[0];
        v57 = *(_DWORD *)(a1 + 136) + 1;
      }
      *(_DWORD *)(a1 + 136) = v57;
      if ( *v39 != -1 )
        --*(_DWORD *)(a1 + 140);
      v58 = *(_DWORD *)(a1 + 208);
      *((_QWORD *)v39 + 1) = 0;
      *v39 = v58;
LABEL_31:
      *((_QWORD *)v39 + 1) = v34;
      v10 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)(a1 + 208) = v10 + 1;
LABEL_19:
      if ( v68 != v70 )
        _libc_free((unsigned __int64)v68);
      return v10;
    case 'N':
LABEL_26:
      v10 = sub_19129F0(a1, v2);
      goto LABEL_19;
    case 'V':
LABEL_25:
      sub_1913290(v71, a1);
LABEL_14:
      v65 = v71[0];
      v66 = v71[1];
      v67 = v72;
      sub_19092D0((__int64)&v68, &v73, v17, v18, v19, v20);
      if ( v73 != &v74 )
        _libc_free((unsigned __int64)v73);
      v21 = sub_1911DB0(a1, (__int64)&v65);
      v22 = *(_DWORD *)(a1 + 24);
      v23 = v21;
      v10 = v21;
      if ( v22 )
      {
        v24 = v64;
        v25 = *(_QWORD *)(a1 + 8);
        v26 = (v22 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
        v27 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( v64 == *v27 )
          goto LABEL_18;
        v46 = 1;
        v47 = 0;
        while ( v28 != -8 )
        {
          if ( !v47 && v28 == -16 )
            v47 = v27;
          v26 = (v22 - 1) & (v46 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v64 == *v27 )
          {
LABEL_18:
            *((_DWORD *)v27 + 2) = v23;
            goto LABEL_19;
          }
          ++v46;
        }
        v48 = *(_DWORD *)(a1 + 16);
        if ( v47 )
          v27 = v47;
        ++*(_QWORD *)a1;
        v49 = v48 + 1;
        if ( 4 * (v48 + 1) < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a1 + 20) - v49 > v22 >> 3 )
          {
LABEL_52:
            *(_DWORD *)(a1 + 16) = v49;
            if ( *v27 != -8 )
              --*(_DWORD *)(a1 + 20);
            *v27 = v24;
            *((_DWORD *)v27 + 2) = 0;
            goto LABEL_18;
          }
LABEL_57:
          sub_177C7D0(a1, v22);
          sub_190E590(a1, &v64, v71);
          v27 = (__int64 *)v71[0];
          v24 = v64;
          v49 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_52;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      v22 *= 2;
      goto LABEL_57;
    default:
      if ( v4 )
      {
        v6 = v4 - 1;
LABEL_23:
        v29 = v6 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v30 = (__int64 *)(v5 + 16LL * v29);
        v31 = *v30;
        if ( v2 == *v30 )
        {
LABEL_24:
          v10 = *(_DWORD *)(a1 + 208);
          *((_DWORD *)v30 + 2) = v10;
          *(_DWORD *)(a1 + 208) = v10 + 1;
          goto LABEL_19;
        }
        v50 = 1;
        v51 = 0;
        while ( v31 != -8 )
        {
          if ( !v51 && v31 == -16 )
            v51 = v30;
          v29 = v6 & (v50 + v29);
          v30 = (__int64 *)(v5 + 16LL * v29);
          v31 = *v30;
          if ( v2 == *v30 )
            goto LABEL_24;
          ++v50;
        }
        v52 = *(_DWORD *)(a1 + 16);
        if ( v51 )
          v30 = v51;
        ++*(_QWORD *)a1;
        v53 = v52 + 1;
        if ( 4 * v53 < 3 * v4 )
        {
          if ( v4 - *(_DWORD *)(a1 + 20) - v53 > v4 >> 3 )
          {
LABEL_64:
            *(_DWORD *)(a1 + 16) = v53;
            if ( *v30 != -8 )
              --*(_DWORD *)(a1 + 20);
            *v30 = v2;
            *((_DWORD *)v30 + 2) = 0;
            goto LABEL_24;
          }
LABEL_69:
          sub_177C7D0(a1, v4);
          sub_190E590(a1, &v64, v71);
          v30 = (__int64 *)v71[0];
          v2 = v64;
          v53 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_64;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      v4 *= 2;
      goto LABEL_69;
  }
}
