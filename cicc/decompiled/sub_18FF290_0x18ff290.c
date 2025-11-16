// Function: sub_18FF290
// Address: 0x18ff290
//
__int64 __fastcall sub_18FF290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 *v7; // rax
  _QWORD *v8; // rdx
  bool v9; // al
  unsigned int v10; // eax
  unsigned int v11; // ebx
  __int64 v12; // r14
  __int64 v13; // r13
  int v14; // r12d
  __int64 *v15; // rcx
  int v16; // eax
  __int64 v17; // r12
  _QWORD *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 *v21; // rbx
  __int64 v22; // rdx
  int v23; // r8d
  int v24; // r9d
  char v25; // di
  int v26; // eax
  __int64 v27; // rax
  __int64 *v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r12
  int v36; // eax
  unsigned __int64 v37; // rax
  __int64 v38; // rdi
  __int64 *v39; // rax
  char v40; // dl
  __int64 v41; // rax
  __int64 *v42; // rsi
  __int64 *v43; // rcx
  int v45; // eax
  int v46; // r12d
  __int64 v47; // rbx
  unsigned int v48; // eax
  __int64 v49; // r12
  __int64 v50; // r15
  __int64 v51; // r14
  __int64 *v52; // rbx
  __int64 *v53; // rax
  __int64 v54; // rbx
  unsigned int v55; // eax
  __int64 v56; // r12
  __int64 *v57; // rbx
  __int64 v58; // [rsp+8h] [rbp-118h]
  __int64 *v59; // [rsp+18h] [rbp-108h]
  unsigned int v60; // [rsp+20h] [rbp-100h]
  int v61; // [rsp+20h] [rbp-100h]
  int v62; // [rsp+20h] [rbp-100h]
  int v63; // [rsp+24h] [rbp-FCh]
  int v64; // [rsp+24h] [rbp-FCh]
  int v65; // [rsp+24h] [rbp-FCh]
  __int64 *v66; // [rsp+28h] [rbp-F8h]
  __int64 *v67; // [rsp+28h] [rbp-F8h]
  __int64 *v68; // [rsp+28h] [rbp-F8h]
  __int64 *v69; // [rsp+30h] [rbp-F0h]
  unsigned int v70; // [rsp+30h] [rbp-F0h]
  unsigned int v71; // [rsp+30h] [rbp-F0h]
  unsigned int v72; // [rsp+30h] [rbp-F0h]
  __int64 v73; // [rsp+38h] [rbp-E8h]
  int v76; // [rsp+58h] [rbp-C8h]
  unsigned __int8 v77; // [rsp+5Fh] [rbp-C1h]
  _QWORD v78[2]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD *v79; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+78h] [rbp-A8h]
  _QWORD v81[4]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v82; // [rsp+A0h] [rbp-80h] BYREF
  __int64 *v83; // [rsp+A8h] [rbp-78h]
  __int64 *v84; // [rsp+B0h] [rbp-70h]
  __int64 v85; // [rsp+B8h] [rbp-68h]
  int v86; // [rsp+C0h] [rbp-60h]
  _BYTE v87[88]; // [rsp+C8h] [rbp-58h] BYREF

  v5 = a1;
  if ( a4 == *(_QWORD *)(a3 - 24) )
  {
    v53 = (__int64 *)sub_157E9C0(a4);
    v73 = sub_159C4F0(v53);
  }
  else
  {
    v7 = (__int64 *)sub_157E9C0(a4);
    v73 = sub_159C540(v7);
  }
  v8 = v81;
  v9 = a4 != *(_QWORD *)(a3 - 24);
  v79 = v81;
  v82 = 0;
  v76 = v9 + 26;
  v83 = (__int64 *)v87;
  v84 = (__int64 *)v87;
  v80 = 0x400000001LL;
  v85 = 4;
  v86 = 0;
  v81[0] = a2;
  v77 = 0;
  v58 = a1 + 88;
  v59 = (__int64 *)(a1 + 136);
  v10 = 1;
  while ( 1 )
  {
    v11 = *(_DWORD *)(v5 + 112);
    v12 = *(_QWORD *)(v5 + 120);
    v13 = v8[v10 - 1];
    LODWORD(v80) = v10 - 1;
    if ( v11 )
    {
      v20 = *(_QWORD *)(v5 + 96);
      v60 = v11 - 1;
      v63 = 1;
      v70 = (v11 - 1) & sub_18FDEE0(v13);
      v66 = 0;
      while ( 1 )
      {
        v21 = (__int64 *)(v20 + 16LL * v70);
        if ( (unsigned __int8)sub_18FB980(v13, *v21) )
        {
          v18 = *(_QWORD **)(v5 + 128);
          v17 = v21[1];
          v15 = v21;
          v19 = *(_QWORD *)(v12 + 16);
          if ( !v18 )
            goto LABEL_11;
          goto LABEL_15;
        }
        if ( *v21 == -8 )
          break;
        if ( *v21 == -16 )
        {
          if ( *v21 == -8 )
            break;
          if ( !v66 )
          {
            if ( *v21 != -16 )
              v21 = 0;
            v66 = v21;
          }
        }
        v70 = v60 & (v63 + v70);
        ++v63;
      }
      v15 = (__int64 *)(v20 + 16LL * v70);
      v11 = *(_DWORD *)(v5 + 112);
      if ( v66 )
        v15 = v66;
      v45 = *(_DWORD *)(v5 + 104);
      ++*(_QWORD *)(v5 + 88);
      v16 = v45 + 1;
      if ( 4 * v16 >= 3 * v11 )
        goto LABEL_6;
      if ( v11 - (v16 + *(_DWORD *)(v5 + 108)) > v11 >> 3 )
        goto LABEL_8;
      sub_18FE1A0(v58, v11);
      v46 = *(_DWORD *)(v5 + 112);
      v15 = 0;
      if ( v46 )
      {
        v47 = *(_QWORD *)(v5 + 96);
        v61 = v46 - 1;
        v64 = 1;
        v48 = (v46 - 1) & sub_18FDEE0(v13);
        v49 = v5;
        v50 = v12;
        v71 = v48;
        v51 = v47;
        v67 = 0;
        while ( 1 )
        {
          v52 = (__int64 *)(v51 + 16LL * v71);
          if ( (unsigned __int8)sub_18FB980(v13, *v52) )
          {
            v12 = v50;
            v15 = v52;
            v5 = v49;
            goto LABEL_7;
          }
          if ( *v52 == -8 )
            break;
          if ( *v52 == -16 )
          {
            if ( *v52 == -8 )
              break;
            if ( !v67 )
            {
              if ( *v52 != -16 )
                v52 = 0;
              v67 = v52;
            }
          }
          v71 = v61 & (v64 + v71);
          ++v64;
        }
        v15 = (__int64 *)(v51 + 16LL * v71);
        v12 = v50;
        v5 = v49;
        if ( v67 )
          v15 = v67;
      }
    }
    else
    {
      ++*(_QWORD *)(v5 + 88);
LABEL_6:
      sub_18FE1A0(v58, 2 * v11);
      v14 = *(_DWORD *)(v5 + 112);
      v15 = 0;
      if ( v14 )
      {
        v54 = *(_QWORD *)(v5 + 96);
        v62 = v14 - 1;
        v65 = 1;
        v55 = (v14 - 1) & sub_18FDEE0(v13);
        v56 = v54;
        v72 = v55;
        v68 = 0;
        while ( 1 )
        {
          v57 = (__int64 *)(v56 + 16LL * v72);
          if ( (unsigned __int8)sub_18FB980(v13, *v57) )
          {
            v15 = (__int64 *)(v56 + 16LL * v72);
            goto LABEL_7;
          }
          if ( *v57 == -8 )
            break;
          if ( *v57 == -16 )
          {
            if ( *v57 == -8 )
              break;
            if ( !v68 )
            {
              if ( *v57 != -16 )
                v57 = 0;
              v68 = v57;
            }
          }
          v72 = v62 & (v65 + v72);
          ++v65;
        }
        v15 = (__int64 *)(v56 + 16LL * v72);
        if ( v68 )
          v15 = v68;
      }
    }
LABEL_7:
    v16 = *(_DWORD *)(v5 + 104) + 1;
LABEL_8:
    *(_DWORD *)(v5 + 104) = v16;
    if ( *v15 != -8 )
      --*(_DWORD *)(v5 + 108);
    *v15 = v13;
    v17 = 0;
    v15[1] = 0;
    v18 = *(_QWORD **)(v5 + 128);
    v19 = *(_QWORD *)(v12 + 16);
    if ( v18 )
    {
LABEL_15:
      *(_QWORD *)(v5 + 128) = *v18;
    }
    else
    {
LABEL_11:
      v69 = v15;
      v18 = (_QWORD *)sub_145CBF0(v59, 32, 8);
      v15 = v69;
    }
    v18[2] = v13;
    *v18 = v19;
    v18[3] = v73;
    v18[1] = v17;
    v15[1] = (__int64)v18;
    *(_QWORD *)(v12 + 16) = v18;
    v22 = *(_QWORD *)(v5 + 16);
    v78[0] = a5;
    v78[1] = a4;
    v25 = v77;
    if ( (unsigned int)sub_1AEC470(v13, v73, v22, v78) )
      v25 = 1;
    v26 = *(unsigned __int8 *)(v13 + 16);
    v77 = v25;
    if ( (unsigned int)(v26 - 35) <= 0x11 && v76 == v26 - 24 )
    {
      v27 = 3LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      {
        v28 = *(__int64 **)(v13 - 8);
        v29 = (__int64)&v28[v27];
      }
      else
      {
        v29 = v13;
        v28 = (__int64 *)(v13 - v27 * 8);
      }
      while ( (__int64 *)v29 != v28 )
      {
        v35 = *v28;
        v36 = *(unsigned __int8 *)(*v28 + 16);
        if ( (unsigned __int8)v36 > 0x17u )
        {
          if ( (_BYTE)v36 == 78 )
          {
            if ( !(unsigned __int8)sub_1560260((_QWORD *)(v35 + 56), -1, 36) )
            {
              if ( *(char *)(v35 + 23) < 0 )
              {
                v30 = sub_1648A40(v35);
                v32 = v30 + v31;
                v33 = 0;
                if ( *(char *)(v35 + 23) < 0 )
                  v33 = sub_1648A40(v35);
                if ( (unsigned int)((v32 - v33) >> 4) )
                  goto LABEL_31;
              }
              v34 = *(_QWORD *)(v35 - 24);
              if ( *(_BYTE *)(v34 + 16) )
                goto LABEL_31;
              v78[0] = *(_QWORD *)(v34 + 112);
              if ( !(unsigned __int8)sub_1560260(v78, -1, 36) )
                goto LABEL_31;
            }
            if ( !*(_BYTE *)(*(_QWORD *)v35 + 8LL) )
              goto LABEL_31;
            v39 = v83;
            if ( v84 != v83 )
              goto LABEL_38;
          }
          else
          {
            v37 = (unsigned int)(v36 - 35);
            if ( (unsigned __int8)v37 > 0x34u )
              goto LABEL_31;
            v38 = 0x1F133FFE23FFFFLL;
            if ( !_bittest64(&v38, v37) )
              goto LABEL_31;
            v39 = v83;
            if ( v84 != v83 )
              goto LABEL_38;
          }
          v42 = &v39[HIDWORD(v85)];
          if ( v39 == v42 )
          {
LABEL_74:
            if ( HIDWORD(v85) >= (unsigned int)v85 )
            {
LABEL_38:
              sub_16CCBA0((__int64)&v82, v35);
              if ( !v40 )
                goto LABEL_31;
            }
            else
            {
              ++HIDWORD(v85);
              *v42 = v35;
              ++v82;
            }
LABEL_39:
            v41 = (unsigned int)v80;
            if ( (unsigned int)v80 >= HIDWORD(v80) )
            {
              sub_16CD150((__int64)&v79, v81, 0, 8, v23, v24);
              v41 = (unsigned int)v80;
            }
            v79[v41] = v35;
            LODWORD(v80) = v80 + 1;
            goto LABEL_31;
          }
          v43 = 0;
          while ( v35 != *v39 )
          {
            if ( *v39 == -2 )
              v43 = v39;
            if ( v42 == ++v39 )
            {
              if ( !v43 )
                goto LABEL_74;
              *v43 = v35;
              --v86;
              ++v82;
              goto LABEL_39;
            }
          }
        }
LABEL_31:
        v28 += 3;
      }
    }
    v10 = v80;
    if ( !(_DWORD)v80 )
      break;
    v8 = v79;
  }
  if ( v84 != v83 )
    _libc_free((unsigned __int64)v84);
  if ( v79 != v81 )
    _libc_free((unsigned __int64)v79);
  return v77;
}
