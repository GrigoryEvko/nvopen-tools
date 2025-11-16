// Function: sub_31B45E0
// Address: 0x31b45e0
//
__int64 __fastcall sub_31B45E0(__int64 a1)
{
  __int64 v1; // r15
  __int64 *v2; // r13
  int v3; // r12d
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // r14
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rax
  _QWORD *v16; // rdx
  _QWORD *v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 *v21; // rcx
  __int64 *v22; // r13
  __int64 *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // r15
  _QWORD *v26; // r13
  unsigned __int64 v27; // r12
  bool v28; // al
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 *v33; // rsi
  __int64 v34; // rdx
  __m128i v35; // kr10_16
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r12
  __int64 v39; // r13
  bool v40; // al
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r13
  _QWORD *v45; // rax
  __int64 v46; // rax
  __int64 v47; // r8
  _QWORD *v48; // r15
  _QWORD *v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rax
  _QWORD *v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rax
  __int64 v55; // r9
  __int64 v56; // r13
  __int64 v57; // rbx
  _QWORD *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rbx
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  __int64 *v70; // [rsp+8h] [rbp-138h]
  __int64 v72; // [rsp+18h] [rbp-128h]
  __int64 *v73; // [rsp+20h] [rbp-120h]
  __int64 v74; // [rsp+28h] [rbp-118h]
  __int64 v75; // [rsp+30h] [rbp-110h]
  _QWORD *v76; // [rsp+40h] [rbp-100h]
  unsigned int v77; // [rsp+40h] [rbp-100h]
  int v78; // [rsp+48h] [rbp-F8h]
  char v79; // [rsp+4Fh] [rbp-F1h]
  __int64 v80; // [rsp+58h] [rbp-E8h]
  int v81; // [rsp+60h] [rbp-E0h]
  __int64 v82; // [rsp+68h] [rbp-D8h]
  char v83; // [rsp+68h] [rbp-D8h]
  __m128i v84; // [rsp+70h] [rbp-D0h] BYREF
  char v85; // [rsp+80h] [rbp-C0h]
  char v86; // [rsp+81h] [rbp-BFh]
  __int64 v87; // [rsp+88h] [rbp-B8h]
  __m128i v88; // [rsp+90h] [rbp-B0h]
  __int64 v89; // [rsp+A0h] [rbp-A0h]
  __int64 v90; // [rsp+A8h] [rbp-98h]
  _QWORD v91[4]; // [rsp+B0h] [rbp-90h] BYREF
  char v92; // [rsp+D0h] [rbp-70h]
  char v93; // [rsp+D1h] [rbp-6Fh]
  _QWORD *v94; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v95; // [rsp+E8h] [rbp-58h]
  _QWORD v96[10]; // [rsp+F0h] [rbp-50h] BYREF

  v1 = 0;
  v73 = *(__int64 **)(a1 + 104);
  v70 = &v73[*(unsigned int *)(a1 + 112)];
  if ( v73 == v70 )
    return 0;
  do
  {
    v2 = *(__int64 **)(*v73 + 16);
    v82 = *(unsigned int *)(*v73 + 24);
    v3 = *(_DWORD *)(*v73 + 24);
    v4 = *(_QWORD *)(*v73 + 8);
    v81 = *(_DWORD *)(*v73 + 128);
    if ( *(_DWORD *)(*v73 + 72) )
      v5 = sub_318B4F0(**(_QWORD **)(*v73 + 64));
    else
      v5 = sub_318B4F0(*v2);
    v7 = v5;
    switch ( *(_DWORD *)(v4 + 8) )
    {
      case 0:
        if ( !v81 )
          return 0;
        v1 = sub_31B1000(a1, v2, v82, v5);
        goto LABEL_6;
      case 1:
        v10 = *v2;
        v94 = v96;
        v95 = 0x200000000LL;
        v11 = *(_DWORD *)(v10 + 32);
        if ( v11 == 11 )
        {
          v64 = sub_318B650(v10);
        }
        else
        {
          if ( v11 != 12 )
          {
            v12 = *v73;
            v13 = *(__int64 **)(*v73 + 136);
            v14 = (__int64)&v13[*(unsigned int *)(*v73 + 144)];
            if ( v13 == (__int64 *)v14 )
            {
              v21 = v96;
            }
            else
            {
              v15 = *v13;
              v16 = v96;
              v17 = v13 + 1;
              v18 = v14;
              v19 = *(_QWORD *)(v15 + 200);
              v20 = 0;
              while ( 1 )
              {
                v16[v20] = v19;
                v20 = (unsigned int)(v95 + 1);
                LODWORD(v95) = v95 + 1;
                if ( (_QWORD *)v18 == v17 )
                  break;
                v19 = *(_QWORD *)(*v17 + 200LL);
                if ( v20 + 1 > (unsigned __int64)HIDWORD(v95) )
                {
                  sub_C8D5F0((__int64)&v94, v96, v20 + 1, 8u, v14, v6);
                  v20 = (unsigned int)v95;
                }
                v16 = v94;
                ++v17;
              }
              v21 = v94;
              v12 = *v73;
            }
LABEL_23:
            v1 = (__int64)sub_31B0F40(a1, *(__int64 **)(v12 + 16), *(unsigned int *)(v12 + 24), v21);
            if ( v1 )
              sub_31B3E50(a1, v2, v82);
            goto LABEL_25;
          }
          v63 = *(_QWORD *)(**(_QWORD **)(*v73 + 136) + 200LL);
          LODWORD(v95) = 1;
          v96[0] = v63;
          v64 = sub_318B6A0(v10);
        }
        v67 = v64;
        v68 = (unsigned int)v95;
        v69 = (unsigned int)v95 + 1LL;
        if ( v69 > HIDWORD(v95) )
        {
          sub_C8D5F0((__int64)&v94, v96, v69, 8u, v65, v66);
          v68 = (unsigned int)v95;
        }
        v94[v68] = v67;
        v21 = v94;
        LODWORD(v95) = v95 + 1;
        v12 = *v73;
        goto LABEL_23;
      case 2:
        v1 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 200LL);
        goto LABEL_6;
      case 3:
        v1 = (__int64)sub_31B0F60(a1, *(_QWORD *)(*(_QWORD *)(v4 + 16) + 200LL), v4 + 24, v5);
        goto LABEL_6;
      case 4:
        v22 = sub_318EB80(*v2);
        if ( (unsigned int)*(unsigned __int8 *)(*v22 + 8) - 17 <= 1 )
        {
          v23 = sub_318E560(v22);
          v24 = *v22;
          v22 = v23;
          v3 *= *(_DWORD *)(v24 + 32);
        }
        v76 = sub_318E570((__int64)v22, v3);
        v94 = v96;
        v95 = 0x400000000LL;
        v25 = *(_QWORD **)(v4 + 16);
        v26 = &v25[2 * *(unsigned int *)(v4 + 24)];
        if ( v26 == v25 )
        {
          v33 = v96;
          v34 = 0;
        }
        else
        {
          do
          {
            v27 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v25 & 4) == 0 )
              v27 = *(_QWORD *)(v27 + 200);
            v28 = sub_318B630(v27);
            if ( v27 && v28 )
            {
              v31 = (unsigned int)v95;
              v32 = (unsigned int)v95 + 1LL;
              if ( v32 > HIDWORD(v95) )
              {
                sub_C8D5F0((__int64)&v94, v96, v32, 8u, v29, v30);
                v31 = (unsigned int)v95;
              }
              v94[v31] = v27;
              LODWORD(v95) = v95 + 1;
            }
            v25 += 2;
          }
          while ( v26 != v25 );
          v33 = v94;
          v34 = (unsigned int)v95;
        }
        sub_31AFE90(&v84, v33, v34, v7);
        v35 = v84;
        v83 = v85;
        v79 = v86;
        v80 = v87;
        v36 = sub_371B680(v76);
        v37 = *(_QWORD *)(v4 + 16);
        v38 = *(_QWORD *)(v36 + 24);
        v1 = v36;
        v72 = v37 + 16LL * *(unsigned int *)(v4 + 24);
        if ( v72 == v37 )
          goto LABEL_25;
        v74 = *(_QWORD *)(v4 + 16);
        v77 = 0;
        break;
      default:
        goto LABEL_6;
    }
    do
    {
      v39 = *(_QWORD *)v74 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v74 & 4) == 0 )
      {
        v56 = *(_QWORD *)(v39 + 200);
        v57 = *(int *)(v74 + 8);
        v58 = sub_318E530(v38);
        v59 = sub_371B620(v58, v57, 0);
        v93 = 1;
        v92 = 3;
        v91[0] = "VExt";
        v60 = *(_QWORD *)(v56 + 24);
        v88 = v35;
        LOBYTE(v89) = v83;
        BYTE1(v89) = v79;
        v90 = v80;
        v39 = (__int64)sub_318BDB0(v56, v59, v60, (__int64)v91, v61, v62, v35.m128i_i64[0], v35.m128i_i64[1], v89);
      }
      v40 = sub_318B630(v39);
      if ( v39 && v40 && (*(_DWORD *)(v39 + 8) != 37 || sub_318B6C0(v39)) )
      {
        v41 = v39;
        if ( sub_318B670(v39) )
        {
          v41 = sub_318B680(v39);
        }
        else if ( *(_DWORD *)(v39 + 8) == 37 )
        {
          v41 = sub_318B6C0(v39);
        }
        v42 = *sub_318EB80(v41);
        if ( *(_BYTE *)(v42 + 8) != 17 )
        {
LABEL_58:
          v53 = sub_318E530(v38);
          v54 = sub_371B620(v53, v77, 0);
          v91[0] = "VIns";
          v93 = 1;
          v88 = v35;
          v92 = 3;
          LOBYTE(v89) = v83;
          BYTE1(v89) = v79;
          v90 = v80;
          ++v77;
          v1 = (__int64)sub_318BC00(v1, v39, v54, v38, (__int64)v91, v55, v35.m128i_i64[0], v35.m128i_i64[1], v89);
          goto LABEL_55;
        }
      }
      else
      {
        v42 = *sub_318EB80(v39);
        if ( *(_BYTE *)(v42 + 8) != 17 )
          goto LABEL_58;
      }
      v78 = *(_DWORD *)(v42 + 32);
      if ( v78 == 1 )
        goto LABEL_58;
      if ( v78 )
      {
        v43 = 0;
        v75 = v39;
        v44 = v1;
        do
        {
          v45 = sub_318E530(v38);
          v46 = sub_371B620(v45, v43, 0);
          v93 = 1;
          v91[0] = "VExt";
          v90 = v80;
          LOBYTE(v89) = v83;
          BYTE1(v89) = v79;
          v88 = v35;
          v92 = 3;
          v48 = sub_318BDB0(v75, v46, v38, (__int64)v91, v47, v80, v35.m128i_i64[0], v35.m128i_i64[1], v89);
          v49 = sub_318E530(v38);
          v50 = (unsigned int)v43++ + v77;
          v51 = sub_371B620(v49, v50, 0);
          v93 = 1;
          v88 = v35;
          BYTE1(v89) = v79;
          LOBYTE(v89) = v83;
          v90 = v80;
          v91[0] = "VIns";
          v92 = 3;
          v52 = sub_318BC00(v44, (__int64)v48, v51, v38, (__int64)v91, v80, v35.m128i_i64[0], v35.m128i_i64[1], v89);
          v44 = (__int64)v52;
        }
        while ( v78 != v43 );
        v77 += v78;
        v1 = (__int64)v52;
      }
LABEL_55:
      v74 += 16;
    }
    while ( v72 != v74 );
LABEL_25:
    if ( v94 != v96 )
      _libc_free((unsigned __int64)v94);
LABEL_6:
    if ( v1 )
    {
      *(_BYTE *)(a1 + 40) = 1;
      *(_QWORD *)(*v73 + 200) = v1;
    }
    ++v73;
  }
  while ( v70 != v73 );
  return v1;
}
