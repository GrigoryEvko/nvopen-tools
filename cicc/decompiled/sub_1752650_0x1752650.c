// Function: sub_1752650
// Address: 0x1752650
//
_QWORD *__fastcall sub_1752650(
        __int64 *a1,
        __int64 ***a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *result; // rax
  __int64 v12; // r14
  char v13; // al
  __int64 v14; // rax
  __int64 **v15; // r13
  unsigned int v16; // r12d
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // kr00_8
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 **v26; // rcx
  __int64 ***v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // r13d
  __int64 *v31; // rax
  __int64 v32; // r12
  const char *v33; // rax
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // r13
  _WORD *v38; // rcx
  __int64 v39; // rdx
  int v40; // esi
  __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // r11
  __int64 *v44; // r10
  __int64 v45; // r15
  unsigned int v46; // esi
  __int64 v47; // rdx
  __int64 v48; // rax
  _WORD *v49; // r12
  _QWORD *v50; // rbx
  __int64 v51; // rdi
  __int64 v52; // rdi
  unsigned __int8 *v53; // rax
  __int64 *v54; // r12
  unsigned __int8 *v55; // rax
  _BYTE *v56; // rax
  char v57; // al
  __int64 v58; // rdi
  unsigned __int8 *v59; // rax
  __int64 *v60; // r12
  unsigned __int8 *v61; // rax
  __int64 *v62; // rsi
  __int64 v63; // rdx
  int v64; // edi
  __int64 v65; // rdi
  unsigned __int8 *v66; // rax
  __int64 v67; // rdi
  __int64 *v68; // r12
  unsigned __int8 *v69; // rax
  _BYTE *v70; // rax
  char v71; // al
  unsigned __int8 *v72; // rax
  __int64 v73; // rdi
  __int64 v74; // r12
  unsigned __int8 *v75; // rax
  __int64 v76; // rdi
  _QWORD *v77; // rax
  __int64 v78; // rdi
  __int64 *v79; // r12
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // rax
  unsigned int v83; // [rsp+8h] [rbp-E8h]
  unsigned int v84; // [rsp+10h] [rbp-E0h]
  __int64 *v85; // [rsp+10h] [rbp-E0h]
  __int64 v86; // [rsp+18h] [rbp-D8h]
  unsigned int v87; // [rsp+20h] [rbp-D0h]
  __int64 v88; // [rsp+20h] [rbp-D0h]
  __int64 v89; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v90; // [rsp+28h] [rbp-C8h]
  _WORD *v91; // [rsp+28h] [rbp-C8h]
  _QWORD *v92; // [rsp+28h] [rbp-C8h]
  _QWORD *v93; // [rsp+28h] [rbp-C8h]
  __int64 v94; // [rsp+28h] [rbp-C8h]
  unsigned int v95; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v96; // [rsp+28h] [rbp-C8h]
  __int64 **v97; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int8 *v98; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD v99[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD *v100; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v101; // [rsp+60h] [rbp-90h]
  _WORD *v102; // [rsp+70h] [rbp-80h] BYREF
  __int64 v103; // [rsp+78h] [rbp-78h]
  _WORD v104[56]; // [rsp+80h] [rbp-70h] BYREF

  result = (_QWORD *)sub_174B490(a1, (__int64)a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( result )
    return result;
  v12 = (__int64)*(a2 - 3);
  v97 = *a2;
  v13 = *(_BYTE *)(v12 + 16);
  if ( (unsigned __int8)(v13 - 35) > 0x11u )
  {
LABEL_11:
    if ( v13 != 78 )
      return sub_174B990(a2, a1[1]);
    v22 = *(_QWORD *)(v12 - 24);
    if ( *(_BYTE *)(v22 + 16) || (*(_BYTE *)(v22 + 33) & 0x20) == 0 )
      return sub_174B990(a2, a1[1]);
    v23 = *(_DWORD *)(v22 + 36);
    if ( v23 != 140 )
    {
      if ( v23 > 0x8C )
      {
        if ( v23 > 0xBC )
        {
          if ( v23 != 206 )
            return sub_174B990(a2, a1[1]);
        }
        else if ( v23 <= 0xBA )
        {
          return sub_174B990(a2, a1[1]);
        }
      }
      else if ( v23 != 8 && v23 - 96 > 1 )
      {
        return sub_174B990(a2, a1[1]);
      }
    }
    v24 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
    v25 = *(_QWORD *)(v24 + 8);
    if ( v25 )
    {
      if ( !*(_QWORD *)(v25 + 8) )
      {
        v26 = v97;
        if ( v23 == 96 || *(_BYTE *)(v24 + 16) == 68 && (v27 = *(__int64 ****)(v24 - 24), v26 = *v27, v97 == *v27) )
        {
          v28 = a1[1];
          v104[0] = 257;
          v90 = sub_1708970(v28, 43, v24, v26, (__int64 *)&v102);
          v29 = *(_QWORD *)(v12 - 24);
          if ( *(_BYTE *)(v29 + 16) )
            BUG();
          v30 = *(_DWORD *)(v29 + 36);
          v31 = (__int64 *)sub_15F2050((__int64)a2);
          v32 = sub_15E26F0(v31, v30, (__int64 *)&v97, 1);
          v102 = v104;
          v103 = 0x100000000LL;
          sub_1752100(v12, (__int64)&v102);
          v33 = sub_1649960(v12);
          v34 = (unsigned int)v103;
          v99[0] = v33;
          v35 = (__int64)v102;
          v101 = 261;
          v99[1] = v36;
          v100 = v99;
          v98 = v90;
          v37 = *(_QWORD *)(*(_QWORD *)v32 + 24LL);
          v38 = &v102[28 * (unsigned int)v103];
          if ( v102 == v38 )
          {
            v42 = sub_1648AB0(72, 2u, 16 * (int)v103);
            if ( v42 )
            {
              v43 = v34;
              v44 = (__int64 *)v35;
              v45 = (__int64)v42;
              v46 = 0;
LABEL_31:
              v85 = v44;
              v86 = v43;
              v88 = (__int64)v42;
              sub_15F1EA0((__int64)v42, **(_QWORD **)(v37 + 16), 54, (__int64)&v42[-3 * v46 - 6], v46 + 2, 0);
              *(_QWORD *)(v88 + 56) = 0;
              sub_15F5B40(v88, v37, v32, (__int64 *)&v98, 1, (__int64)&v100, v85, v86);
              v48 = v88;
LABEL_32:
              v92 = (_QWORD *)v48;
              sub_15F2500(v45, v12);
              v49 = v102;
              result = v92;
              v50 = &v102[28 * (unsigned int)v103];
              if ( v102 != (_WORD *)v50 )
              {
                do
                {
                  v51 = *(v50 - 3);
                  v50 -= 7;
                  if ( v51 )
                    j_j___libc_free_0(v51, v50[6] - v51);
                  if ( (_QWORD *)*v50 != v50 + 2 )
                    j_j___libc_free_0(*v50, v50[2] + 1LL);
                }
                while ( v49 != (_WORD *)v50 );
                v49 = v102;
                result = v92;
              }
              if ( v49 != v104 )
              {
                v93 = result;
                _libc_free((unsigned __int64)v49);
                return v93;
              }
              return result;
            }
          }
          else
          {
            v39 = (__int64)v102;
            v40 = 0;
            do
            {
              v41 = *(_QWORD *)(v39 + 40) - *(_QWORD *)(v39 + 32);
              v39 += 56;
              v40 += v41 >> 3;
            }
            while ( v38 != (_WORD *)v39 );
            v91 = &v102[28 * (unsigned int)v103];
            v42 = sub_1648AB0(72, v40 + 2, 16 * (int)v103);
            if ( v42 )
            {
              v43 = v34;
              v44 = (__int64 *)v35;
              v45 = (__int64)v42;
              v46 = 0;
              do
              {
                v47 = *(_QWORD *)(v35 + 40) - *(_QWORD *)(v35 + 32);
                v35 += 56;
                v46 += v47 >> 3;
              }
              while ( v91 != (_WORD *)v35 );
              goto LABEL_31;
            }
          }
          v45 = 0;
          v48 = 0;
          goto LABEL_32;
        }
      }
    }
    return sub_174B990(a2, a1[1]);
  }
  v14 = *(_QWORD *)(v12 + 8);
  if ( !v14 || *(_QWORD *)(v14 + 8) )
    return sub_174B990(a2, a1[1]);
  v15 = (__int64 **)sub_174A3D0(*(_BYTE **)(v12 - 48));
  v89 = sub_174A3D0(*(_BYTE **)(v12 - 24));
  v84 = sub_16431F0(*(_QWORD *)v12);
  v16 = sub_16431F0((__int64)v15);
  v17 = sub_16431F0(v89);
  v87 = v17;
  if ( v16 >= v17 )
    v17 = v16;
  v83 = v17;
  v18 = (unsigned int)sub_16431F0((__int64)v97);
  v20 = v19;
  v21 = v87;
  switch ( *(_BYTE *)(v12 + 16) )
  {
    case '$':
    case '&':
      if ( 2 * (int)v18 + 1 > v84 || (unsigned int)v18 < v83 )
        goto LABEL_9;
      v52 = a1[1];
      v104[0] = 257;
      v53 = sub_1708970(v52, 43, *(_QWORD *)(v12 - 48), v97, (__int64 *)&v102);
      v104[0] = 257;
      v54 = (__int64 *)v53;
      v55 = sub_1708970(a1[1], 43, *(_QWORD *)(v12 - 24), v97, (__int64 *)&v102);
      v104[0] = 257;
      v94 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v12 + 16) - 24, v54, (__int64)v55, (__int64)&v102, 0);
      sub_15F2500(v94, v12);
      return (_QWORD *)v94;
    case '(':
      v21 = v16 + v87;
      if ( (unsigned int)v21 > v84 || (unsigned int)v18 < v83 )
        goto LABEL_9;
      v65 = a1[1];
      v104[0] = 257;
      v66 = sub_1708970(v65, 43, *(_QWORD *)(v12 - 48), v97, (__int64 *)&v102);
      v67 = a1[1];
      v104[0] = 257;
      v68 = (__int64 *)v66;
      v69 = sub_1708970(v67, 43, *(_QWORD *)(v12 - 24), v97, (__int64 *)&v102);
      v104[0] = 257;
      v63 = (__int64)v69;
      v62 = v68;
      v64 = 16;
      goto LABEL_57;
    case '+':
      if ( unk_4FA1E20 || !LOBYTE(qword_4FA1F40[20]) )
        goto LABEL_9;
      v95 = v18;
      v56 = sub_16D40F0((__int64)qword_4FBB490);
      v18 = v95;
      v57 = v56 ? *v56 : LOBYTE(qword_4FBB490[2]);
      if ( v57 || 2 * v95 > v84 || v95 < v83 )
        goto LABEL_9;
      v58 = a1[1];
      v104[0] = 257;
      v59 = sub_1708970(v58, 43, *(_QWORD *)(v12 - 48), v97, (__int64 *)&v102);
      v104[0] = 257;
      v60 = (__int64 *)v59;
      v61 = sub_1708970(a1[1], 43, *(_QWORD *)(v12 - 24), v97, (__int64 *)&v102);
      v62 = v60;
      v104[0] = 257;
      v63 = (__int64)v61;
      v64 = 19;
      goto LABEL_57;
    case '.':
      if ( unk_4FA1E20 || !LOBYTE(qword_4FA1F40[20]) )
        goto LABEL_9;
      v70 = sub_16D40F0((__int64)qword_4FBB490);
      v21 = v87;
      v71 = v70 ? *v70 : LOBYTE(qword_4FBB490[2]);
      if ( v84 == v83 || v71 )
        goto LABEL_9;
      v104[0] = 257;
      if ( v16 < v87 )
        v15 = (__int64 **)v89;
      v72 = sub_1708970(a1[1], 43, *(_QWORD *)(v12 - 48), v15, (__int64 *)&v102);
      v73 = a1[1];
      v104[0] = 257;
      v74 = (__int64)v72;
      v75 = sub_1708970(v73, 43, *(_QWORD *)(v12 - 24), v15, (__int64 *)&v102);
      v76 = a1[1];
      v104[0] = 257;
      v77 = sub_174BAA0(v76, v74, (__int64)v75, v12, (__int64 *)&v102, *(double *)a3.m128_u64, a4, a5);
      v104[0] = 257;
      return (_QWORD *)sub_15FE110(v77, (__int64)v97, (__int64)&v102, 0);
    default:
      v21 = v20;
LABEL_9:
      if ( !sub_15FB6D0(v12, 0, v21, v18) )
      {
        v12 = (__int64)*(a2 - 3);
        v13 = *(_BYTE *)(v12 + 16);
        goto LABEL_11;
      }
      v78 = a1[1];
      v104[0] = 257;
      v79 = (__int64 *)sub_1708970(v78, 43, *(_QWORD *)(v12 - 24), v97, (__int64 *)&v102);
      v82 = sub_15A1390(*v79, 43, v80, v81);
      v63 = (__int64)v79;
      v62 = (__int64 *)v82;
      v64 = 14;
      v104[0] = 257;
LABEL_57:
      v96 = (unsigned __int8 *)sub_15FB440(v64, v62, v63, (__int64)&v102, 0);
      sub_15F2530(v96, v12, 1);
      result = v96;
      break;
  }
  return result;
}
