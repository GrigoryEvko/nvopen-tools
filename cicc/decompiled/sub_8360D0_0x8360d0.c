// Function: sub_8360D0
// Address: 0x8360d0
//
_QWORD *__fastcall sub_8360D0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        _BOOL4 a6,
        const __m128i *a7,
        int a8,
        unsigned int a9,
        unsigned int a10,
        int a11,
        int a12,
        int a13,
        int a14,
        unsigned int a15,
        int a16,
        int a17,
        __int64 *a18,
        __int64 a19,
        _DWORD *a20,
        _DWORD *a21)
{
  __int64 *v23; // rbx
  __int64 v24; // r10
  __int64 v25; // r12
  char v26; // al
  char v27; // al
  __int64 v28; // rax
  char v29; // al
  __int64 v30; // r15
  __int64 v31; // rdx
  unsigned int v32; // eax
  int v33; // eax
  int v34; // eax
  _QWORD *result; // rax
  char v36; // al
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // r8
  __int64 v41; // rcx
  __int64 v42; // rdi
  int v43; // eax
  __int64 k; // rax
  _QWORD *v45; // rax
  int v46; // eax
  char v47; // al
  char v48; // al
  char v49; // al
  __int64 v50; // rax
  __int64 i; // rdx
  int v52; // eax
  char v53; // r12
  __int64 v54; // r15
  __int64 v55; // rbx
  _QWORD *v56; // rax
  __int64 v57; // rax
  __int64 j; // rax
  char v59; // dl
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // [rsp+0h] [rbp-D0h]
  __int64 v63; // [rsp+8h] [rbp-C8h]
  __int64 v64; // [rsp+8h] [rbp-C8h]
  __int64 v65; // [rsp+10h] [rbp-C0h]
  __int64 v66; // [rsp+10h] [rbp-C0h]
  __int64 v67; // [rsp+10h] [rbp-C0h]
  __int64 v68; // [rsp+10h] [rbp-C0h]
  _QWORD *v70; // [rsp+28h] [rbp-A8h]
  int v71; // [rsp+34h] [rbp-9Ch]
  int v73; // [rsp+48h] [rbp-88h]
  __int64 v74; // [rsp+50h] [rbp-80h]
  _BOOL4 v76; // [rsp+60h] [rbp-70h]
  int v77; // [rsp+64h] [rbp-6Ch]
  __int64 v78; // [rsp+68h] [rbp-68h]
  _BOOL4 v79; // [rsp+68h] [rbp-68h]
  __int64 v80; // [rsp+68h] [rbp-68h]
  __int64 v81; // [rsp+78h] [rbp-58h] BYREF
  _BYTE v82[80]; // [rsp+80h] [rbp-50h] BYREF
  __int64 *v83; // [rsp+138h] [rbp+68h]

  v23 = a18;
  v76 = a6;
  v70 = (_QWORD *)*a18;
  v74 = 0;
  v24 = sub_82C1B0(a1, (__int64)a18, a19, (__int64)v82);
  if ( !v24 || a8 | a6 | a15 )
  {
LABEL_3:
    if ( a17 != 3 )
    {
LABEL_4:
      v77 = a17 == 1;
      goto LABEL_5;
    }
    goto LABEL_70;
  }
  v80 = v24;
  v48 = sub_877F80(v24);
  v24 = v80;
  if ( v48 != 1 )
  {
    v49 = sub_877F80(v80);
    v24 = v80;
    if ( v49 != 7 )
    {
      v50 = sub_8792C0(v80);
      v24 = v80;
      for ( i = v50; *(_BYTE *)(v50 + 140) == 12; v50 = *(_QWORD *)(v50 + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(v50 + 168) + 40LL) )
        goto LABEL_79;
      v57 = **(_QWORD **)(i + 168);
      if ( v57 && (*(_BYTE *)(v57 + 35) & 1) != 0 && qword_4F04C50 )
      {
        for ( j = *(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 152LL);
              *(_BYTE *)(j + 140) == 12;
              j = *(_QWORD *)(j + 160) )
        {
          ;
        }
        if ( *(_QWORD *)(*(_QWORD *)(j + 168) + 40LL) )
          goto LABEL_79;
      }
      v59 = *(_BYTE *)(a1 + 80);
      v60 = a1;
      if ( v59 == 16 )
      {
        v60 = **(_QWORD **)(a1 + 88);
        v59 = *(_BYTE *)(v60 + 80);
      }
      if ( v59 == 24 )
      {
        v60 = *(_QWORD *)(v60 + 88);
        v59 = *(_BYTE *)(v60 + 80);
      }
      if ( v59 == 17 && *(_BYTE *)(v60 + 96) )
      {
LABEL_79:
        v52 = sub_830940(0, &v81);
        v24 = v80;
        v76 = v52;
        if ( v52 )
        {
          v74 = v81;
          v76 = v81 != 0;
        }
        else
        {
          v74 = 0;
        }
        goto LABEL_3;
      }
    }
  }
  v76 = 0;
  v74 = 0;
  if ( a17 != 3 )
    goto LABEL_4;
LABEL_70:
  v77 = 1;
  if ( dword_4F077BC && !(_DWORD)qword_4F077B4 )
    v77 = qword_4F077A8 > 0x9CA3u;
  while ( 1 )
  {
LABEL_5:
    if ( !a5 )
      goto LABEL_62;
    v25 = a1;
    if ( *(_BYTE *)(a1 + 80) == 17 )
      v25 = *(_QWORD *)(a1 + 88);
    v78 = v24;
    v26 = sub_877F80(v25);
    v24 = v78;
    if ( v26 == 1 )
    {
      v79 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 64) + 168LL) + 110LL) & 2) != 0;
      if ( v24 )
        goto LABEL_10;
      v71 = 0;
      v73 = 0;
LABEL_35:
      if ( !v79 )
        goto LABEL_36;
    }
    else
    {
      v47 = sub_877F80(v25);
      v24 = v78;
      if ( v47 != 7 || (v53 = *(_BYTE *)(a1 + 80), v54 = a1, v53 == 17) && (v54 = *(_QWORD *)(a1 + 88)) == 0 )
      {
LABEL_62:
        if ( v24 )
        {
          v79 = 0;
          goto LABEL_10;
        }
        goto LABEL_67;
      }
      v83 = v23;
      v55 = v54;
      while ( 1 )
      {
        v56 = **(_QWORD ***)(sub_8792C0(v55) + 168);
        if ( v56 && (!*v56 || (*(_BYTE *)(*v56 + 32LL) & 4) != 0) )
        {
          if ( (unsigned int)sub_8D3C40(v56[1]) )
            break;
        }
        if ( v53 == 17 )
        {
          v55 = *(_QWORD *)(v55 + 8);
          if ( v55 )
            continue;
        }
        v23 = v83;
        v24 = v78;
        goto LABEL_62;
      }
      v23 = v83;
      v24 = v78;
      if ( v78 )
      {
        v79 = 1;
LABEL_10:
        v71 = 0;
        v73 = 0;
        while ( 2 )
        {
          while ( 2 )
          {
            v29 = *(_BYTE *)(v24 + 80);
            v30 = *v23;
            v31 = v24;
            if ( v29 == 16 )
            {
              v31 = **(_QWORD **)(v24 + 88);
              v29 = *(_BYTE *)(v31 + 80);
            }
            if ( v29 == 24 )
              v31 = *(_QWORD *)(v31 + 88);
            if ( a13 && *(_BYTE *)(v31 + 80) == 20 )
              goto LABEL_15;
            if ( !v79 )
            {
              if ( !a5 )
                goto LABEL_26;
              if ( !a4 )
                goto LABEL_26;
              if ( *(_QWORD *)a4 )
                goto LABEL_26;
              if ( *(_BYTE *)(a4 + 8) != 1 )
                goto LABEL_26;
              v63 = v24;
              v65 = v31;
              v36 = sub_877F80(v31);
              v24 = v63;
              if ( v36 != 1 )
                goto LABEL_26;
              v37 = sub_8792C0(v65);
              v24 = v63;
              v38 = **(_QWORD **)(v37 + 168);
              if ( !v38 )
                goto LABEL_26;
              v62 = v63;
              v66 = *(_QWORD *)(v65 + 64);
              v64 = *(_QWORD *)(v38 + 8);
              v39 = sub_8D32E0(v64);
              v41 = v64;
              v24 = v62;
              if ( v39 )
              {
                v61 = sub_8D46C0(v64);
                v24 = v62;
                v41 = v61;
              }
              while ( *(_BYTE *)(v41 + 140) == 12 )
                v41 = *(_QWORD *)(v41 + 160);
              v42 = v66;
              if ( v66 == v41 || (v67 = v24, v43 = sub_8D97D0(v42, v41, 0, v41, v40), v24 = v67, v43) )
                v32 = 0;
              else
LABEL_26:
                v32 = a10;
              sub_833B90(
                v24,
                a1,
                a2,
                a3,
                0,
                0,
                (_QWORD *)a4,
                v76,
                a7,
                v74,
                a8,
                a9,
                v32,
                a11,
                a12,
                a14,
                a15,
                v77,
                a16,
                a17,
                v23,
                a20,
                a21,
                &v81);
              if ( v30 != *v23 )
                goto LABEL_15;
LABEL_28:
              if ( unk_4F04C48 != -1 && *(char *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 12) < 0 )
                goto LABEL_35;
              v33 = 1;
              if ( (_DWORD)v81 )
                v33 = v71;
              v71 = v33;
              v34 = 1;
              if ( !(_DWORD)v81 )
                v34 = v73;
              v73 = v34;
              v24 = sub_82C230(v82);
              if ( !v24 )
                goto LABEL_35;
              continue;
            }
            break;
          }
          v27 = *(_BYTE *)(v31 + 80);
          if ( (unsigned __int8)(v27 - 10) <= 1u )
          {
            v28 = *(_QWORD *)(v31 + 88);
            if ( (*(_BYTE *)(v28 + 194) & 0x10) == 0 )
            {
LABEL_14:
              if ( *(_BYTE *)(v28 + 174) != 7 )
                goto LABEL_15;
              for ( k = *(_QWORD *)(v28 + 152); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              v45 = **(_QWORD ***)(k + 168);
              if ( !v45 || *v45 && (*(_BYTE *)(*v45 + 32LL) & 4) == 0 )
                goto LABEL_15;
              v68 = v24;
              v46 = sub_8D3C40(v45[1]);
              v24 = v68;
              if ( !v46 )
                goto LABEL_15;
            }
          }
          else
          {
            if ( v27 != 20 )
              goto LABEL_15;
            v28 = *(_QWORD *)(*(_QWORD *)(v31 + 88) + 176LL);
            if ( (*(_BYTE *)(v28 + 194) & 0x10) == 0 )
              goto LABEL_14;
          }
          sub_833B90(
            v24,
            a1,
            a2,
            a3,
            0,
            0,
            a5,
            v76,
            a7,
            v74,
            a8,
            a9,
            0,
            a11,
            a12,
            a14,
            a15,
            v77,
            a16,
            a17,
            v23,
            a20,
            a21,
            &v81);
          if ( v30 != *v23 )
          {
            *(_BYTE *)(*v23 + 145) |= 8u;
LABEL_15:
            v24 = sub_82C230(v82);
            if ( !v24 )
              goto LABEL_35;
            continue;
          }
          goto LABEL_28;
        }
      }
      v73 = 0;
      v71 = 0;
    }
    if ( v70 == (_QWORD *)*v23 )
    {
      v79 = 0;
      v24 = sub_82C1B0(a1, (__int64)v23, a19, (__int64)v82);
      if ( v24 )
        goto LABEL_10;
LABEL_67:
      v73 = 0;
      v71 = 0;
    }
LABEL_36:
    result = (_QWORD *)dword_4F077BC;
    if ( !dword_4F077BC )
      return result;
    result = &qword_4F077A8;
    if ( qword_4F077A8 <= 0x9CA3u )
      return result;
    result = v70;
    if ( v70 != (_QWORD *)*v23 )
      return result;
    result = (_QWORD *)((v77 ^ 1) & v73 & (v71 ^ 1u));
    if ( (((unsigned __int8)v77 ^ 1) & v73 & ((unsigned __int8)v71 ^ 1) & 1) == 0 )
      return result;
    result = (_QWORD *)(a12 | a15);
    v77 = a12 | a15;
    if ( !(a12 | a15) )
      return result;
    v24 = sub_82C1B0(a1, (__int64)v23, a19, (__int64)v82);
  }
}
