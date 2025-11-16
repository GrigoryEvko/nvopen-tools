// Function: sub_2590AF0
// Address: 0x2590af0
//
__int64 __fastcall sub_2590AF0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // r12
  int v8; // r14d
  __int64 v10; // rax
  __int64 i; // rax
  __int64 v12; // r13
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rcx
  unsigned int v16; // edx
  unsigned int v17; // eax
  __int64 v18; // r8
  unsigned __int8 *v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // r9d
  unsigned int v22; // eax
  unsigned int v23; // r13d
  _QWORD *v24; // r8
  unsigned int v25; // eax
  __int64 v26; // r13
  char *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r12
  char *v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // ecx
  bool v34; // zf
  int v36; // r9d
  int v37; // r8d
  unsigned int v38; // eax
  __int64 v39; // rcx
  __int64 v40; // rax
  unsigned int v41; // ecx
  __int64 v42; // r13
  char v43; // al
  unsigned __int8 *v44; // rax
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // r13
  _QWORD *v47; // rax
  __int64 j; // rbx
  __int64 v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  char v52; // di
  unsigned __int64 v53; // rax
  unsigned int v54; // edx
  __int64 v55; // rdi
  unsigned int v56; // r15d
  unsigned int v57; // r13d
  __int64 v58; // r12
  __int64 v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned int v63; // ecx
  unsigned int v64; // [rsp+0h] [rbp-150h]
  __int64 v65; // [rsp+8h] [rbp-148h]
  unsigned int v66; // [rsp+8h] [rbp-148h]
  unsigned __int8 *v67; // [rsp+8h] [rbp-148h]
  __int64 v68; // [rsp+8h] [rbp-148h]
  _QWORD *v69; // [rsp+8h] [rbp-148h]
  _QWORD *v70; // [rsp+8h] [rbp-148h]
  unsigned int v71; // [rsp+18h] [rbp-138h]
  unsigned int v72; // [rsp+18h] [rbp-138h]
  __int64 v73; // [rsp+18h] [rbp-138h]
  unsigned int v74; // [rsp+18h] [rbp-138h]
  __int64 v75; // [rsp+18h] [rbp-138h]
  char *v76; // [rsp+18h] [rbp-138h]
  _QWORD *v77; // [rsp+20h] [rbp-130h]
  unsigned int v79; // [rsp+30h] [rbp-120h]
  unsigned __int64 v80; // [rsp+30h] [rbp-120h]
  __int64 v81; // [rsp+30h] [rbp-120h]
  char v83; // [rsp+46h] [rbp-10Ah] BYREF
  char v84; // [rsp+47h] [rbp-109h] BYREF
  __int64 v85; // [rsp+48h] [rbp-108h] BYREF
  __int64 v86; // [rsp+50h] [rbp-100h] BYREF
  bool v87; // [rsp+58h] [rbp-F8h]
  __m128i v88[3]; // [rsp+60h] [rbp-F0h] BYREF
  char v89; // [rsp+90h] [rbp-C0h]
  char v90[8]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v91; // [rsp+A8h] [rbp-A8h]
  unsigned int v92; // [rsp+B8h] [rbp-98h]
  __int64 v93; // [rsp+C8h] [rbp-88h]
  __int64 v94; // [rsp+D0h] [rbp-80h]
  __int64 v95; // [rsp+D8h] [rbp-78h]
  char v96[8]; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v97; // [rsp+E8h] [rbp-68h]
  unsigned int v98; // [rsp+F8h] [rbp-58h]
  __int64 v99; // [rsp+108h] [rbp-48h]
  __int64 v100; // [rsp+110h] [rbp-40h]
  __int64 v101; // [rsp+118h] [rbp-38h]

  v7 = v90;
  v8 = 0;
  v10 = sub_2568740(a3, a4);
  sub_254C700((__int64)v90, v10);
  sub_254C700((__int64)v96, a3 + 200);
  for ( i = 0; *(_DWORD *)(a5 + 40) > (unsigned int)i; v8 = i )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(a5 + 32) + 8 * i);
    v13 = *(_QWORD *)(v12 + 24);
    if ( *(_BYTE *)v13 > 0x1Cu )
    {
      v14 = v13 & 0xFFFFFFFFFFFFFFFBLL;
      v15 = v13 | 4;
      if ( !v92 )
        goto LABEL_29;
      v16 = v92 - 1;
      v17 = (v92 - 1) & (v15 ^ (v15 >> 9));
      v18 = *(_QWORD *)(v91 + 8LL * v17);
      if ( v15 != v18 )
      {
        v36 = 1;
        while ( v18 != -4 )
        {
          v17 = v16 & (v36 + v17);
          v18 = *(_QWORD *)(v91 + 8LL * v17);
          if ( v15 == v18 )
            goto LABEL_5;
          ++v36;
        }
        v37 = 1;
        v38 = v16 & (v14 ^ (v14 >> 9));
        v39 = *(_QWORD *)(v91 + 8LL * v38);
        if ( v14 != v39 )
        {
          while ( v39 != -4 )
          {
            v38 = v16 & (v37 + v38);
            v39 = *(_QWORD *)(v91 + 8LL * v38);
            if ( v14 == v39 )
              goto LABEL_5;
            ++v37;
          }
LABEL_29:
          v40 = v93;
          while ( v99 != v40 || v94 != v100 || v95 != v101 )
          {
            v40 = sub_3106C80(v7);
            v93 = v40;
            if ( v13 == v40 )
              goto LABEL_5;
          }
          goto LABEL_23;
        }
      }
LABEL_5:
      v83 = 0;
      v84 = 0;
      v19 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
      v20 = sub_258FD00(a2, a1, v19, (char *)v12, (unsigned __int8 *)v13, &v83, &v84);
      v21 = v20;
      v77 = (_QWORD *)(a6 + 32);
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v12 + 8LL) + 8LL) != 14 )
        goto LABEL_6;
      v72 = v20;
      v65 = *(_QWORD *)v12;
      sub_D66840(v88, (_BYTE *)v13);
      v21 = v72;
      if ( !v89 )
        goto LABEL_6;
      v73 = v65;
      if ( v65 == v88[0].m128i_i64[0]
        && (v42 = v88[0].m128i_i64[1], v88[0].m128i_i64[1] >= 0)
        && (v66 = v21, v43 = sub_B46560((unsigned __int8 *)v13), v21 = v66, !v43) )
      {
        v44 = sub_25536C0(v73, &v85, *(_QWORD *)(a2[26] + 104LL), 1);
        v21 = v66;
        if ( v44
          && (v74 = v66, v67 = v44, v45 = sub_250D070((_QWORD *)(a1 + 72)), v21 = v74, v67 == (unsigned __int8 *)v45) )
        {
          v86 = v42 & 0x3FFFFFFFFFFFFFFFLL;
          v87 = (v42 & 0x4000000000000000LL) != 0;
          v46 = a6 + 32;
          v80 = sub_CA1930(&v86);
          v21 = v74;
          v47 = *(_QWORD **)(a6 + 40);
          if ( !v47 )
            goto LABEL_61;
          do
          {
            if ( v85 > v47[4] )
            {
              v47 = (_QWORD *)v47[3];
            }
            else
            {
              v46 = (unsigned __int64)v47;
              v47 = (_QWORD *)v47[2];
            }
          }
          while ( v47 );
          if ( (_QWORD *)v46 == v77 || v85 < *(_QWORD *)(v46 + 32) )
          {
LABEL_61:
            v64 = v74;
            v75 = v85;
            v68 = v46;
            v49 = sub_22077B0(0x30u);
            *(_QWORD *)(v49 + 40) = 0;
            v46 = v49;
            *(_QWORD *)(v49 + 32) = v75;
            v50 = sub_2567530((_QWORD *)(a6 + 24), v68, (__int64 *)(v49 + 32));
            if ( v51 )
            {
              v52 = v77 == v51 || v50 || v75 < v51[4];
              sub_220F040(v52, v46, v51, v77);
              v21 = v64;
              ++*(_QWORD *)(a6 + 64);
            }
            else
            {
              v70 = v50;
              j_j___libc_free_0(v46);
              v21 = v64;
              v46 = (unsigned __int64)v70;
            }
          }
          v53 = v80;
          if ( *(_QWORD *)(v46 + 40) >= v80 )
            v53 = *(_QWORD *)(v46 + 40);
          *(_QWORD *)(v46 + 40) = v53;
          v54 = *(_DWORD *)(a6 + 16);
          v24 = *(_QWORD **)(a6 + 48);
          v23 = v54;
          if ( v77 != v24 )
          {
            v81 = a5;
            v55 = *(_QWORD *)(a6 + 48);
            v56 = *(_DWORD *)(a6 + 16);
            v57 = v21;
            v76 = v7;
            v58 = v13;
            v59 = v56;
            do
            {
              v62 = *(_QWORD *)(v55 + 32);
              if ( v62 > v59 )
                break;
              v60 = *(_QWORD *)(v55 + 40) + v62;
              v69 = v24;
              if ( v59 < v60 )
                v59 = v60;
              v61 = sub_220EEE0(v55);
              v24 = v69;
              v55 = v61;
            }
            while ( v77 != (_QWORD *)v61 );
            v54 = v59;
            v21 = v57;
            v13 = v58;
            v23 = v56;
            v7 = v76;
            a5 = v81;
          }
          v63 = v54;
          if ( *(_DWORD *)(a6 + 20) >= v54 )
            v63 = *(_DWORD *)(a6 + 20);
          if ( v23 < v54 )
            v23 = v54;
          v22 = v63;
        }
        else
        {
          v23 = *(_DWORD *)(a6 + 16);
          v22 = *(_DWORD *)(a6 + 20);
          v24 = *(_QWORD **)(a6 + 48);
        }
      }
      else
      {
LABEL_6:
        v22 = *(_DWORD *)(a6 + 20);
        v23 = *(_DWORD *)(a6 + 16);
        v24 = *(_QWORD **)(a6 + 48);
      }
      if ( v21 >= v22 )
        v22 = v21;
      v71 = v22;
      *(_DWORD *)(a6 + 20) = v22;
      v25 = v23;
      if ( v21 >= v23 )
        v25 = v21;
      v79 = v25;
      v26 = v25;
      *(_DWORD *)(a6 + 16) = v25;
      if ( v77 != v24 )
      {
        v27 = v7;
        v28 = (__int64)v24;
        v29 = v26;
        v30 = v27;
        do
        {
          v32 = *(_QWORD *)(v28 + 32);
          if ( v29 < v32 )
          {
            v33 = v29;
            v7 = v30;
            v25 = v33;
            if ( v79 >= v33 )
              v33 = v79;
            v79 = v33;
            goto LABEL_20;
          }
          v31 = *(_QWORD *)(v28 + 40) + v32;
          if ( v29 < v31 )
            v29 = v31;
          v28 = sub_220EEE0(v28);
        }
        while ( a6 + 32 != v28 );
        v41 = v29;
        v7 = v30;
        v25 = v41;
        if ( v79 >= v41 )
          v41 = v79;
        v79 = v41;
      }
LABEL_20:
      if ( v25 < v71 )
        v25 = v71;
      v34 = v84 == 0;
      *(_DWORD *)(a6 + 20) = v25;
      *(_DWORD *)(a6 + 16) = v79;
      if ( !v34 )
      {
        for ( j = *(_QWORD *)(v13 + 16); j; j = *(_QWORD *)(j + 8) )
        {
          v88[0].m128i_i64[0] = j;
          sub_25789E0(a5, v88[0].m128i_i64);
        }
      }
    }
LABEL_23:
    i = (unsigned int)(v8 + 1);
  }
  sub_C7D6A0(v97, 8LL * v98, 8);
  return sub_C7D6A0(v91, 8LL * v92, 8);
}
