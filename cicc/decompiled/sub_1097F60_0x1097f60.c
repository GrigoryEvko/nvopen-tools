// Function: sub_1097F60
// Address: 0x1097f60
//
__int64 __fastcall sub_1097F60(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  unsigned int v7; // r9d
  int v8; // ebx
  char v9; // si
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  size_t v14; // rax
  size_t v15; // rax
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rax
  __m128i si128; // xmm0
  _BYTE *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  _BYTE *v25; // r15
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  char *v32; // rcx
  char *v33; // rax
  char v34; // dl
  __int64 v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  _BYTE *v50; // rax
  __int64 v51; // rcx
  _BYTE *v52; // rax
  __int64 v53; // rdx
  char *v54; // rax
  __int64 v55; // rcx
  char v56; // dl
  __int64 v57; // rax
  _BYTE *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  _BYTE *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  _BYTE *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  _BYTE *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // [rsp+18h] [rbp-88h] BYREF
  __int64 *v73; // [rsp+20h] [rbp-80h] BYREF
  __int64 v74; // [rsp+28h] [rbp-78h]
  __int64 v75; // [rsp+30h] [rbp-70h] BYREF
  __int64 v76; // [rsp+38h] [rbp-68h]
  unsigned int v77; // [rsp+40h] [rbp-60h]
  int v78; // [rsp+48h] [rbp-58h] BYREF
  __int64 v79; // [rsp+50h] [rbp-50h]
  __int64 v80; // [rsp+58h] [rbp-48h]
  __int64 v81; // [rsp+60h] [rbp-40h]
  unsigned int v82; // [rsp+68h] [rbp-38h]

  *(_QWORD *)(a2 + 104) = *(_QWORD *)(a2 + 152);
  v3 = sub_1095C70((_QWORD *)a2);
  v8 = v3;
  if ( *(_BYTE *)(a2 + 178) != 1 && v3 == 35 && *(_BYTE *)(a2 + 177) )
  {
    v11 = *(_QWORD *)a2;
    LODWORD(v73) = 0;
    v74 = 0;
    v75 = 0;
    v77 = 1;
    v76 = 0;
    v78 = 0;
    v79 = 0;
    v80 = 0;
    v82 = 1;
    v81 = 0;
    v12 = (*(__int64 (__fastcall **)(__int64, __int64 **, __int64, __int64))(v11 + 32))(a2, &v73, 2, 1);
    if ( *(_BYTE *)(a2 + 176) && v12 == 2 && (_DWORD)v73 == 4 && v78 == 3 )
    {
      *(_QWORD *)(a2 + 152) = *(_QWORD *)(a2 + 104);
      v22 = sub_1097DC0((_QWORD *)a2);
      *(_BYTE *)(a2 + 115) = 0;
      v24 = v23;
      v25 = v22;
      sub_EAA0A0(a2 + 8, *(_QWORD *)(a2 + 8), (unsigned __int64)&v78, v26, v27, v28);
      *(_BYTE *)(a2 + 115) = 0;
      sub_EAA0A0(a2 + 8, *(_QWORD *)(a2 + 8), (unsigned __int64)&v73, v29, v30, v31);
      *(_DWORD *)a1 = 8;
      *(_QWORD *)(a1 + 8) = v25;
      *(_QWORD *)(a1 + 16) = v24;
      *(_DWORD *)(a1 + 32) = 64;
      *(_QWORD *)(a1 + 24) = 0;
      goto LABEL_33;
    }
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 144) + 64LL) )
    {
      sub_1096300(a1, a2);
LABEL_33:
      if ( v82 > 0x40 && v81 )
        j_j___libc_free_0_0(v81);
      if ( v77 > 0x40 && v76 )
        j_j___libc_free_0_0(v76);
      return a1;
    }
    if ( v82 > 0x40 && v81 )
      j_j___libc_free_0_0(v81);
    if ( v77 > 0x40 && v76 )
      j_j___libc_free_0_0(v76);
  }
  if ( (unsigned __int8)sub_1097E30(a2, *(const char **)(a2 + 104), v4, v5, v6, v7) )
  {
    sub_1096300(a1, a2);
  }
  else if ( sub_1097E90(a2, *(const char **)(a2 + 104)) )
  {
    v13 = *(_QWORD *)(a2 + 144);
    v14 = strlen(*(const char **)(v13 + 40));
    *(_WORD *)(a2 + 176) = 257;
    *(_QWORD *)(a2 + 152) += v14 - 1;
    v15 = strlen(*(const char **)(v13 + 40));
    v16 = *(_QWORD *)(a2 + 104);
    *(_DWORD *)a1 = 9;
    *(_QWORD *)(a1 + 16) = v15;
    *(_QWORD *)(a1 + 8) = v16;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  else
  {
    v9 = *(_BYTE *)(a2 + 177);
    if ( v8 == -1 )
    {
      v17 = *(_BYTE *)(a2 + 179);
      if ( v9 || !v17 )
      {
        *(_WORD *)(a2 + 176) = v17 != 0 ? 0x101 : 0;
        v18 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 0;
        *(_QWORD *)(a1 + 8) = v18;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
      }
      else
      {
        v19 = *(_QWORD *)(a2 + 104);
        *(_WORD *)(a2 + 176) = 257;
        *(_DWORD *)a1 = 9;
        *(_QWORD *)(a1 + 8) = v19;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
      }
    }
    else
    {
      *(_WORD *)(a2 + 176) = 0;
      switch ( v8 )
      {
        case 0:
        case 9:
        case 32:
          v32 = *(char **)(a2 + 152);
          *(_BYTE *)(a2 + 177) = v9;
          if ( *v32 == 9 || *v32 == 32 )
          {
            v33 = v32 + 1;
            do
            {
              do
              {
                *(_QWORD *)(a2 + 152) = v33;
                v34 = *v33;
                v32 = v33++;
              }
              while ( v34 == 32 );
            }
            while ( v34 == 9 );
          }
          if ( *(_BYTE *)(a2 + 112) )
          {
            (**(void (__fastcall ***)(__int64, __int64))a2)(a1, a2);
          }
          else
          {
            v71 = *(_QWORD *)(a2 + 104);
            *(_DWORD *)a1 = 11;
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 8) = v71;
            *(_QWORD *)(a1 + 16) = &v32[-v71];
            *(_QWORD *)(a1 + 24) = 0;
          }
          break;
        case 10:
          *(_WORD *)(a2 + 176) = 257;
          v66 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 9;
          *(_QWORD *)(a1 + 8) = v66;
          *(_QWORD *)(a1 + 16) = 1;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 13:
          v64 = *(_BYTE **)(a2 + 152);
          *(_WORD *)(a2 + 176) = 257;
          if ( v64 != (_BYTE *)(*(_QWORD *)(a2 + 160) + *(_QWORD *)(a2 + 168)) && *v64 == 10 )
            *(_QWORD *)(a2 + 152) = ++v64;
          v65 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 9;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 8) = v65;
          *(_QWORD *)(a1 + 16) = &v64[-v65];
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 33:
          v69 = *(_BYTE **)(a2 + 152);
          v70 = *(_QWORD *)(a2 + 104);
          if ( *v69 == 61 )
          {
            *(_QWORD *)(a2 + 152) = v69 + 1;
            *(_DWORD *)a1 = 36;
            *(_QWORD *)(a1 + 8) = v70;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 35;
            *(_QWORD *)(a1 + 8) = v70;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 34:
          sub_1097BF0(a1, a2);
          break;
        case 35:
          if ( *(_BYTE *)(*(_QWORD *)(a2 + 144) + 22LL) )
            goto LABEL_105;
          v68 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 38;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v68;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 36:
          if ( *(_BYTE *)(a2 + 119) && word_3F64060[**(unsigned __int8 **)(a2 + 152)] != 0xFFFF )
            goto LABEL_29;
          if ( *(_BYTE *)(*(_QWORD *)(a2 + 144) + 182LL) )
            goto LABEL_105;
          v67 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 27;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v67;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 37:
          if ( *(_BYTE *)(a2 + 119) && (unsigned __int8)(**(_BYTE **)(a2 + 152) - 48) <= 1u )
            goto LABEL_29;
          v60 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 37;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v60;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 38:
          v58 = *(_BYTE **)(a2 + 152);
          v59 = *(_QWORD *)(a2 + 104);
          if ( *v58 == 38 )
          {
            *(_QWORD *)(a2 + 152) = v58 + 1;
            *(_DWORD *)a1 = 34;
            *(_QWORD *)(a1 + 8) = v59;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 33;
            *(_QWORD *)(a1 + 8) = v59;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 39:
          sub_10978A0(a1, a2);
          break;
        case 40:
          v57 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 17;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v57;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 41:
          v44 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 18;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v44;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 42:
          v43 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 24;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v43;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 43:
          v42 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 12;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v42;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 44:
          v41 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 26;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v41;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 45:
          v61 = *(_BYTE **)(a2 + 152);
          v62 = *(_QWORD *)(a2 + 104);
          if ( *v61 == 62 )
          {
            *(_QWORD *)(a2 + 152) = v61 + 1;
            *(_DWORD *)a1 = 47;
            *(_QWORD *)(a1 + 8) = v62;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 13;
            *(_QWORD *)(a1 + 8) = v62;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 47:
          *(_BYTE *)(a2 + 177) = v9;
          sub_1096450(a1, a2);
          break;
        case 48:
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 55:
        case 56:
        case 57:
LABEL_29:
          sub_1096640(a1, a2);
          break;
        case 58:
          v63 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 10;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v63;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 60:
          v54 = *(char **)(a2 + 152);
          v55 = *(_QWORD *)(a2 + 104);
          v56 = *v54;
          if ( *v54 == 61 )
          {
            *(_QWORD *)(a2 + 152) = v54 + 1;
            *(_DWORD *)a1 = 40;
            *(_QWORD *)(a1 + 8) = v55;
            *(_QWORD *)(a1 + 16) = 2;
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 24) = 0;
          }
          else if ( v56 == 62 )
          {
            *(_QWORD *)(a2 + 152) = v54 + 1;
            *(_DWORD *)a1 = 42;
            *(_QWORD *)(a1 + 8) = v55;
            *(_QWORD *)(a1 + 16) = 2;
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 24) = 0;
          }
          else
          {
            if ( v56 == 60 )
            {
              *(_QWORD *)(a2 + 152) = v54 + 1;
              *(_DWORD *)a1 = 41;
              *(_QWORD *)(a1 + 8) = v55;
              *(_QWORD *)(a1 + 16) = 2;
            }
            else
            {
              *(_DWORD *)a1 = 39;
              *(_QWORD *)(a1 + 8) = v55;
              *(_QWORD *)(a1 + 16) = 1;
            }
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 24) = 0;
          }
          break;
        case 61:
          v52 = *(_BYTE **)(a2 + 152);
          v53 = *(_QWORD *)(a2 + 104);
          if ( *v52 == 61 )
          {
            *(_QWORD *)(a2 + 152) = v52 + 1;
            *(_DWORD *)a1 = 29;
            *(_QWORD *)(a1 + 8) = v53;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 28;
            *(_QWORD *)(a1 + 8) = v53;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 62:
          v50 = *(_BYTE **)(a2 + 152);
          v51 = *(_QWORD *)(a2 + 104);
          if ( *v50 == 61 )
          {
            *(_QWORD *)(a2 + 152) = v50 + 1;
            *(_DWORD *)a1 = 44;
            *(_QWORD *)(a1 + 8) = v51;
            *(_QWORD *)(a1 + 16) = 2;
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 24) = 0;
          }
          else
          {
            if ( *v50 == 62 )
            {
              *(_QWORD *)(a2 + 152) = v50 + 1;
              *(_DWORD *)a1 = 45;
              *(_QWORD *)(a1 + 8) = v51;
              *(_QWORD *)(a1 + 16) = 2;
            }
            else
            {
              *(_DWORD *)a1 = 43;
              *(_QWORD *)(a1 + 8) = v51;
              *(_QWORD *)(a1 + 16) = 1;
            }
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 24) = 0;
          }
          break;
        case 63:
          if ( *(_BYTE *)(*(_QWORD *)(a2 + 144) + 181LL) )
            goto LABEL_105;
          v49 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 23;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v49;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 64:
          if ( *(_BYTE *)(*(_QWORD *)(a2 + 144) + 183LL) )
            goto LABEL_105;
          v48 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 46;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v48;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 91:
          v47 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 19;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v47;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 92:
          v46 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 16;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v46;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 93:
          v45 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 20;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v45;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 94:
          v40 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 32;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v40;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 123:
          v39 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 21;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v39;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 124:
          v37 = *(_BYTE **)(a2 + 152);
          v38 = *(_QWORD *)(a2 + 104);
          if ( *v37 == 124 )
          {
            *(_QWORD *)(a2 + 152) = v37 + 1;
            *(_DWORD *)a1 = 31;
            *(_QWORD *)(a1 + 8) = v38;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 30;
            *(_QWORD *)(a1 + 8) = v38;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 125:
          v36 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 22;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v36;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        case 126:
          v35 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 14;
          *(_QWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 8) = v35;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          break;
        default:
          if ( isalpha(v8) || v8 == 95 || v8 == 46 )
          {
LABEL_105:
            sub_1096170(a1, a2);
          }
          else
          {
            v72 = 26;
            v73 = &v75;
            v20 = (__m128i *)sub_22409D0(&v73, &v72, 0);
            si128 = _mm_load_si128((const __m128i *)&xmmword_3F90190);
            v73 = (__int64 *)v20;
            v75 = v72;
            qmemcpy(&v20[1], "r in input", 10);
            *v20 = si128;
            v74 = v72;
            *((_BYTE *)v73 + v72) = 0;
            sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)&v73);
            if ( v73 != &v75 )
              j_j___libc_free_0(v73, v75 + 1);
          }
          break;
      }
    }
  }
  return a1;
}
