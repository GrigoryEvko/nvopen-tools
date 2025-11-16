// Function: sub_1807A30
// Address: 0x1807a30
//
__int64 __fastcall sub_1807A30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        _BYTE *a6,
        double a7,
        double a8,
        double a9,
        unsigned int a10,
        unsigned int a11,
        unsigned __int8 a12,
        unsigned __int8 a13,
        unsigned int a14)
{
  __int64 v15; // r12
  __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // r12
  __int64 v19; // r15
  __int64 result; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  unsigned int v23; // ebx
  __int64 v24; // r12
  _QWORD *v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  _QWORD *v28; // r14
  _QWORD *v29; // rax
  unsigned __int8 *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r13
  _QWORD *v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 *v40; // r11
  __int64 v41; // rax
  unsigned __int64 *v42; // rbx
  __int64 v43; // rax
  unsigned __int64 v44; // rcx
  __int64 v45; // rsi
  unsigned __int8 *v46; // rsi
  _QWORD *v47; // rax
  unsigned __int64 *v48; // rbx
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 *v54; // [rsp+10h] [rbp-130h]
  unsigned int v55; // [rsp+18h] [rbp-128h]
  __int64 v56; // [rsp+20h] [rbp-120h]
  int v58; // [rsp+34h] [rbp-10Ch]
  __int64 v59; // [rsp+38h] [rbp-108h]
  __int64 v60; // [rsp+40h] [rbp-100h]
  __int64 *v64; // [rsp+68h] [rbp-D8h] BYREF
  __int64 *v65; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v66; // [rsp+78h] [rbp-C8h]
  __int64 v67[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v68; // [rsp+90h] [rbp-B0h]
  unsigned __int8 *v69[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v70; // [rsp+B0h] [rbp-90h]
  __int64 *v71; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v72; // [rsp+C8h] [rbp-78h]
  unsigned __int64 *v73; // [rsp+D0h] [rbp-70h]
  _QWORD *v74; // [rsp+D8h] [rbp-68h]
  __int64 v75; // [rsp+E0h] [rbp-60h]
  int v76; // [rsp+E8h] [rbp-58h]
  __int64 v77; // [rsp+F0h] [rbp-50h]
  __int64 v78; // [rsp+F8h] [rbp-48h]

  v15 = *(_QWORD *)(*(_QWORD *)a6 + 24LL);
  v16 = v15;
  if ( *(_BYTE *)(v15 + 8) == 16 )
    v16 = **(_QWORD **)(v15 + 16);
  v17 = sub_127FA20(a2, v16);
  v18 = *(_QWORD *)(v15 + 32);
  v19 = 0;
  v55 = (v17 + 7) & 0xFFFFFFF8;
  v54 = (__int64 *)sub_15A0680(a3, 0, 0);
  v56 = (unsigned int)v18;
  result = a13;
  if ( (_DWORD)v18 )
  {
    while ( *(_BYTE *)(a4 + 16) == 8 )
    {
      v21 = v19 - (*(_DWORD *)(a4 + 20) & 0xFFFFFFF);
      result = 3 * v21;
      v22 = *(_QWORD *)(a4 + 24 * v21);
      if ( *(_BYTE *)(v22 + 16) != 13 )
        goto LABEL_49;
      v23 = *(_DWORD *)(v22 + 32);
      if ( v23 <= 0x40 )
      {
        if ( *(_QWORD *)(v22 + 24) )
        {
LABEL_49:
          v24 = a5;
LABEL_8:
          v25 = (_QWORD *)sub_16498A0(v24);
          v71 = 0;
          v74 = v25;
          v75 = 0;
          v76 = 0;
          v77 = 0;
          v78 = 0;
          v72 = *(_QWORD *)(v24 + 40);
          v73 = (unsigned __int64 *)(v24 + 24);
          v26 = *(unsigned __int8 **)(v24 + 48);
          v69[0] = v26;
          if ( v26 )
          {
            sub_1623A60((__int64)v69, (__int64)v26, 2);
            if ( v71 )
              sub_161E7C0((__int64)&v71, (__int64)v71);
            v71 = (__int64 *)v69[0];
            if ( v69[0] )
              sub_1623210((__int64)v69, v69[0], (__int64)&v71);
          }
          v68 = 257;
          v65 = v54;
          v27 = sub_15A0680(a3, v19, 0);
          v66 = v27;
          if ( a6[16] > 0x10u || *((_BYTE *)v65 + 16) > 0x10u || *(_BYTE *)(v27 + 16) > 0x10u )
          {
            v70 = 257;
            v35 = *(_QWORD *)a6;
            if ( *(_BYTE *)(*(_QWORD *)a6 + 8LL) == 16 )
              v35 = **(_QWORD **)(v35 + 16);
            v36 = *(_QWORD *)(v35 + 24);
            v37 = sub_1648A60(72, 3u);
            v28 = v37;
            if ( v37 )
            {
              v60 = (__int64)v37;
              v59 = (__int64)(v37 - 9);
              v38 = *(_QWORD *)a6;
              if ( *(_BYTE *)(*(_QWORD *)a6 + 8LL) == 16 )
                v38 = **(_QWORD **)(v38 + 16);
              v58 = *(_DWORD *)(v38 + 8) >> 8;
              v39 = (__int64 *)sub_15F9F50(v36, (__int64)&v65, 2);
              v40 = (__int64 *)sub_1646BA0(v39, v58);
              v41 = *(_QWORD *)a6;
              if ( *(_BYTE *)(*(_QWORD *)a6 + 8LL) == 16
                || (v41 = *v65, *(_BYTE *)(*v65 + 8) == 16)
                || (v41 = *(_QWORD *)v66, *(_BYTE *)(*(_QWORD *)v66 + 8LL) == 16) )
              {
                v40 = sub_16463B0(v40, *(_QWORD *)(v41 + 32));
              }
              sub_15F1EA0((__int64)v28, (__int64)v40, 32, v59, 3, 0);
              v28[7] = v36;
              v28[8] = sub_15F9F50(v36, (__int64)&v65, 2);
              sub_15F9CE0((__int64)v28, (__int64)a6, (__int64 *)&v65, 2, (__int64)v69);
            }
            else
            {
              v60 = 0;
            }
            if ( v72 )
            {
              v42 = v73;
              sub_157E9D0(v72 + 40, (__int64)v28);
              v43 = v28[3];
              v44 = *v42;
              v28[4] = v42;
              v44 &= 0xFFFFFFFFFFFFFFF8LL;
              v28[3] = v44 | v43 & 7;
              *(_QWORD *)(v44 + 8) = v28 + 3;
              *v42 = *v42 & 7 | (unsigned __int64)(v28 + 3);
            }
            sub_164B780(v60, v67);
            if ( v71 )
            {
              v64 = v71;
              sub_1623A60((__int64)&v64, (__int64)v71, 2);
              v45 = v28[6];
              if ( v45 )
                sub_161E7C0((__int64)(v28 + 6), v45);
              v46 = (unsigned __int8 *)v64;
              v28[6] = v64;
              if ( v46 )
                sub_1623210((__int64)&v64, v46, (__int64)(v28 + 6));
            }
          }
          else
          {
            BYTE4(v69[0]) = 0;
            v28 = (_QWORD *)sub_15A2E80(0, (__int64)a6, &v65, 2u, 0, (__int64)v69, 0);
          }
          result = sub_1807990(a1, a5, v24, (__int64)v28, a10, a11, a7, a8, a9, v55, a12, a13, a14);
          if ( v71 )
            result = sub_161E7C0((__int64)&v71, (__int64)v71);
        }
      }
      else
      {
        result = sub_16A57B0(v22 + 24);
        v24 = a5;
        if ( v23 != (_DWORD)result )
          goto LABEL_8;
      }
      if ( v56 == ++v19 )
        return result;
    }
    v29 = (_QWORD *)sub_16498A0(a5);
    v30 = *(unsigned __int8 **)(a5 + 48);
    v71 = 0;
    v74 = v29;
    v31 = *(_QWORD *)(a5 + 40);
    v75 = 0;
    v72 = v31;
    v76 = 0;
    v77 = 0;
    v78 = 0;
    v73 = (unsigned __int64 *)(a5 + 24);
    v69[0] = v30;
    if ( v30 )
    {
      sub_1623A60((__int64)v69, (__int64)v30, 2);
      if ( v71 )
        sub_161E7C0((__int64)&v71, (__int64)v71);
      v71 = (__int64 *)v69[0];
      if ( v69[0] )
        sub_1623210((__int64)v69, v69[0], (__int64)&v71);
    }
    v68 = 257;
    v32 = sub_1643360(v74);
    v33 = sub_159C470(v32, v19, 0);
    if ( *(_BYTE *)(a4 + 16) > 0x10u || *(_BYTE *)(v33 + 16) > 0x10u )
    {
      v70 = 257;
      v47 = sub_1648A60(56, 2u);
      v34 = v47;
      if ( v47 )
        sub_15FA320((__int64)v47, (_QWORD *)a4, v33, (__int64)v69, 0);
      if ( v72 )
      {
        v48 = v73;
        sub_157E9D0(v72 + 40, (__int64)v34);
        v49 = v34[3];
        v50 = *v48;
        v34[4] = v48;
        v50 &= 0xFFFFFFFFFFFFFFF8LL;
        v34[3] = v50 | v49 & 7;
        *(_QWORD *)(v50 + 8) = v34 + 3;
        *v48 = *v48 & 7 | (unsigned __int64)(v34 + 3);
      }
      sub_164B780((__int64)v34, v67);
      if ( v71 )
      {
        v65 = v71;
        sub_1623A60((__int64)&v65, (__int64)v71, 2);
        v51 = v34[6];
        if ( v51 )
          sub_161E7C0((__int64)(v34 + 6), v51);
        v52 = (unsigned __int8 *)v65;
        v34[6] = v65;
        if ( v52 )
          sub_1623210((__int64)&v65, v52, (__int64)(v34 + 6));
      }
    }
    else
    {
      v34 = (_QWORD *)sub_15A37D0((_BYTE *)a4, v33, 0);
    }
    v24 = sub_1AA92B0(v34, a5, 0, 0, 0, 0);
    if ( v71 )
      sub_161E7C0((__int64)&v71, (__int64)v71);
    goto LABEL_8;
  }
  return result;
}
