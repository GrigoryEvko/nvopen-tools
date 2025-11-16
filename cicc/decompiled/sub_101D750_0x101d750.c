// Function: sub_101D750
// Address: 0x101d750
//
__int64 __fastcall sub_101D750(unsigned __int8 *a1, unsigned __int8 *a2, __m128i *a3, int a4)
{
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned __int8 v8; // al
  __int64 v9; // r14
  __int64 v11; // rdx
  bool v12; // zf
  unsigned __int16 v13; // ax
  unsigned __int8 *v14; // rdi
  __int64 *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // r14
  int v22; // eax
  __int64 v23; // rsi
  __int64 *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  _BYTE *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned int v33; // eax
  __int64 v34; // rsi
  unsigned __int16 v35; // ax
  unsigned __int8 *v36; // rax
  unsigned int v37; // eax
  _QWORD *v38; // r8
  int v39; // eax
  int v40; // eax
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 *v43; // rsi
  char v44; // al
  int v45; // eax
  unsigned int v46; // [rsp+10h] [rbp-140h]
  unsigned int v47; // [rsp+14h] [rbp-13Ch]
  int v48; // [rsp+14h] [rbp-13Ch]
  unsigned int v49; // [rsp+18h] [rbp-138h]
  __int64 *v50; // [rsp+18h] [rbp-138h]
  unsigned __int64 v51; // [rsp+18h] [rbp-138h]
  unsigned int v52; // [rsp+30h] [rbp-120h]
  unsigned __int8 *v53; // [rsp+30h] [rbp-120h]
  __int64 v54; // [rsp+38h] [rbp-118h]
  unsigned int v55; // [rsp+38h] [rbp-118h]
  __int64 v56; // [rsp+38h] [rbp-118h]
  unsigned int v57; // [rsp+38h] [rbp-118h]
  __int64 *v58; // [rsp+58h] [rbp-F8h] BYREF
  unsigned __int8 *v59; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v60; // [rsp+68h] [rbp-E8h] BYREF
  __int64 **v61; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v62; // [rsp+78h] [rbp-D8h] BYREF
  __int64 v63; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v64; // [rsp+88h] [rbp-C8h] BYREF
  __int64 *v65; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+98h] [rbp-B8h] BYREF
  __int64 v67[2]; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int64 v68; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned int v69; // [rsp+B8h] [rbp-98h]
  __int64 *v70; // [rsp+C0h] [rbp-90h] BYREF
  unsigned int v71; // [rsp+C8h] [rbp-88h]
  __int64 *v72; // [rsp+D0h] [rbp-80h] BYREF
  __int64 **v73; // [rsp+D8h] [rbp-78h]
  __int64 v74[2]; // [rsp+E0h] [rbp-70h] BYREF
  __int64 *v75; // [rsp+F0h] [rbp-60h] BYREF
  __int64 *v76; // [rsp+F8h] [rbp-58h] BYREF
  __int64 *v77; // [rsp+100h] [rbp-50h] BYREF
  __int64 *v78; // [rsp+108h] [rbp-48h]
  __int64 *v79; // [rsp+110h] [rbp-40h]

  v4 = (__int64)a1;
  v5 = (__int64)a2;
  v8 = *a2;
  if ( *a1 <= 0x15u )
  {
    if ( v8 > 0x15u )
    {
      v8 = *a1;
      v4 = (__int64)a2;
      v5 = (__int64)a1;
    }
    else
    {
      v9 = sub_96E6C0(0x1Cu, (__int64)a1, a2, a3->m128i_i64[0]);
      if ( v9 )
        return v9;
      v8 = *a2;
    }
  }
  if ( v8 == 13 )
    return v5;
  if ( !(unsigned __int8)sub_1003090((__int64)a3, (unsigned __int8 *)v5) )
  {
    if ( v4 == v5 )
      return v4;
    if ( !(unsigned __int8)sub_FFFE90(v5) )
    {
      v75 = 0;
      v9 = v4;
      if ( (unsigned __int8)sub_995B10(&v75, v5) )
        return v9;
      v9 = sub_1010510((char *)v4, (unsigned __int64 **)v5, a3, a4);
      if ( v9 )
        return v9;
      v9 = sub_1010510((char *)v5, (unsigned __int64 **)v4, a3, a4);
      if ( v9 )
        return v9;
      v9 = sub_FFEF50((char *)v4, (_BYTE *)v5, 28);
      if ( v9 )
        return v9;
      LOBYTE(v76) = 0;
      v75 = (__int64 *)&v58;
      if ( !(unsigned __int8)sub_991580((__int64)&v75, v5) )
      {
LABEL_17:
        v72 = &v62;
        if ( (unsigned __int8)sub_1007280(&v72, v5, v11) )
        {
          v12 = *(_BYTE *)v4 == 42;
          v76 = 0;
          v75 = &v63;
          if ( v12 )
          {
            if ( *(_QWORD *)(v4 - 64) )
            {
              v20 = *(_QWORD *)(v4 - 32);
              v63 = *(_QWORD *)(v4 - 64);
              if ( (unsigned __int8)sub_995B10(&v76, v20) )
              {
                if ( (unsigned __int8)sub_9B64A0(
                                        v63,
                                        a3->m128i_i64[0],
                                        0,
                                        0,
                                        a3[2].m128i_i64[0],
                                        a3[2].m128i_i64[1],
                                        a3[1].m128i_i64[1],
                                        1) )
                {
                  sub_9AC330((__int64)&v75, v63, 0, a3);
                  v21 = v62;
                  v55 = *(_DWORD *)(v21 + 8) - sub_9871A0(v62);
                  sub_D95160((__int64)&v72, (__int64)&v75);
                  v22 = sub_9871A0((__int64)&v72);
                  LODWORD(v21) = (_DWORD)v73 - v22;
                  sub_969240((__int64 *)&v72);
                  if ( v55 >= (unsigned int)v21 )
                  {
                    v9 = sub_AD6530(*(_QWORD *)(v5 + 8), (__int64)&v75);
                    sub_969240((__int64 *)&v77);
                    sub_969240((__int64 *)&v75);
                    return v9;
                  }
                  sub_969240((__int64 *)&v77);
                  sub_969240((__int64 *)&v75);
                }
              }
            }
          }
        }
        v9 = sub_1006BA0(a3, (char *)v4, (char *)v5, 1);
        if ( v9 )
          return v9;
        v9 = (__int64)sub_101B370(28, (__int64 *)v4, (__int64 *)v5, a3, a4);
        if ( v9 )
          return v9;
        v9 = (__int64)sub_101C7F0(28, (__int64 *)v4, (unsigned __int8 *)v5, 0x1Du, a3, a4);
        if ( v9 )
          return v9;
        v9 = (__int64)sub_101C7F0(28, (__int64 *)v4, (unsigned __int8 *)v5, 0x1Eu, a3, a4);
        if ( v9 )
          return v9;
        if ( *(_BYTE *)v4 != 86 && *(_BYTE *)v5 != 86 )
          goto LABEL_25;
        if ( !sub_1001970(*(_QWORD *)(v4 + 8), 1) )
          goto LABEL_102;
        v12 = *(_BYTE *)v5 == 86;
        v75 = (__int64 *)v4;
        if ( v12 && (unsigned __int8)sub_FFFFF0(&v75, v5) )
          return v5;
        v12 = *(_BYTE *)v4 == 86;
        v75 = (__int64 *)v5;
        if ( !v12 || !(unsigned __int8)sub_FFFFF0(&v75, v4) )
        {
LABEL_102:
          v36 = sub_101C8A0(28, (__int64 *)v4, (__int64 *)v5, a3, a4);
          if ( v36 )
            return (__int64)v36;
LABEL_25:
          if ( *(_BYTE *)v4 == 84 || *(_BYTE *)v5 == 84 )
          {
            v36 = sub_101CAB0(28, (__int64 *)v4, (__int64 *)v5, a3, a4);
            if ( v36 )
              return (__int64)v36;
          }
          if ( !a3[4].m128i_i8[0] )
            goto LABEL_28;
          LOBYTE(v73) = 0;
          v72 = (__int64 *)&v58;
          if ( !(unsigned __int8)sub_991580((__int64)&v72, v5) )
            goto LABEL_28;
          v75 = &v60;
          v76 = (__int64 *)&v59;
          v78 = &v64;
          LOBYTE(v77) = 0;
          v79 = (__int64 *)&v61;
          if ( !sub_100A750((__int64)&v75, 29, (unsigned __int8 *)v4) )
            goto LABEL_28;
          v37 = sub_BCB060(*(_QWORD *)(v4 + 8));
          v57 = v37;
          LODWORD(v38) = v37;
          v47 = *((_DWORD *)v59 + 2);
          if ( v47 > 0x40 )
          {
            v51 = v37;
            v53 = v59;
            v45 = sub_C444A0((__int64)v59);
            LODWORD(v38) = v51;
            if ( v47 - v45 <= 0x40 && v51 > **(_QWORD **)v53 )
              v38 = **(_QWORD ***)v53;
          }
          else if ( (unsigned __int64)v37 > *(_QWORD *)v59 )
          {
            v38 = *(_QWORD **)v59;
          }
          v48 = (int)v38;
          v52 = (unsigned int)v38;
          sub_9AC330((__int64)&v72, (__int64)v61, 0, a3);
          v39 = sub_9871D0((__int64)&v72);
          v46 = (_DWORD)v73 - v39;
          if ( v52 >= (int)v73 - v39 )
          {
            sub_9AC330((__int64)&v75, v60, 0, a3);
            v40 = sub_9871D0((__int64)&v75);
            v49 = (_DWORD)v76 - v40;
            sub_F0A5D0((__int64)v67, v57, v46);
            sub_F0A5D0((__int64)&v70, v57, v49);
            sub_9865C0((__int64)&v68, (__int64)&v70);
            if ( v69 > 0x40 )
            {
              sub_C47690((__int64 *)&v68, v52);
            }
            else
            {
              if ( v48 == v69 )
                v41 = 0;
              else
                v41 = v68 << v48;
              v42 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v69;
              if ( !v69 )
                v42 = 0;
              v68 = v42 & v41;
            }
            sub_969240((__int64 *)&v70);
            v50 = v58;
            v43 = v58;
            if ( sub_10024C0(v67, v58) && (v44 = sub_986FD0((__int64)&v68, v43), v43 = v50, !v44) )
            {
              v9 = (__int64)v61;
            }
            else
            {
              if ( !sub_10024C0((__int64 *)&v68, v43) || sub_986FD0((__int64)v67, v43) )
              {
                sub_969240((__int64 *)&v68);
                sub_969240(v67);
                sub_969240((__int64 *)&v77);
                sub_969240((__int64 *)&v75);
                goto LABEL_91;
              }
              v9 = v64;
            }
            sub_969240((__int64 *)&v68);
            sub_969240(v67);
            sub_969240((__int64 *)&v77);
            sub_969240((__int64 *)&v75);
            sub_969240(v74);
            sub_969240((__int64 *)&v72);
            return v9;
          }
LABEL_91:
          sub_969240(v74);
          sub_969240((__int64 *)&v72);
LABEL_28:
          v76 = (__int64 *)&v65;
          v75 = &v60;
          v77 = &v60;
          v78 = (__int64 *)&v61;
          if ( (unsigned __int8)sub_100A9C0(&v75, 30, (unsigned __int8 *)v4) )
          {
            v12 = *(_BYTE *)v5 == 59;
            v72 = v65;
            v73 = v61;
            if ( v12 )
            {
              v34 = v5;
              if ( (unsigned __int8)sub_FFE640((__int64 *)&v72, v5) )
                return sub_AD6530(*(_QWORD *)(v4 + 8), v34);
            }
          }
          v12 = *(_BYTE *)v4 == 59;
          LOBYTE(v77) = 0;
          v75 = v67;
          v76 = &v66;
          if ( v12 && (unsigned __int8)sub_FFF310((__int64)&v75, v4) )
          {
            v29 = v66;
            sub_9865C0((__int64)&v68, v66);
            sub_987160((__int64)&v68, v29, v30, v31, v32);
            v33 = v69;
            v34 = 30;
            v69 = 0;
            v71 = v33;
            v73 = &v70;
            v70 = (__int64 *)v68;
            v72 = (__int64 *)v67[0];
            if ( sub_100AAA0((__int64)&v72, 30, (unsigned __int8 *)v5) )
            {
              sub_969240((__int64 *)&v70);
              sub_969240((__int64 *)&v68);
              return sub_AD6530(*(_QWORD *)(v4 + 8), v34);
            }
            sub_969240((__int64 *)&v70);
            sub_969240((__int64 *)&v68);
          }
          if ( !sub_1001970(*(_QWORD *)(v4 + 8), 1) )
          {
LABEL_32:
            if ( a4 == 3 )
              return sub_FFE7D0(28, v4, (unsigned __int8 *)v5, a3->m128i_i64[0], a3[2].m128i_i64[1]);
            return v9;
          }
          v13 = sub_9A18B0(v4, (_BYTE *)v5, a3->m128i_i64[0], 1u, 0);
          if ( HIBYTE(v13) )
          {
            if ( !(_BYTE)v13 )
              return sub_AD6450(*(_QWORD *)(v4 + 8));
            return v4;
          }
          v35 = sub_9A18B0(v5, (_BYTE *)v4, a3->m128i_i64[0], 1u, 0);
          if ( !HIBYTE(v35) )
            goto LABEL_32;
          if ( !(_BYTE)v35 )
            return sub_AD6450(*(_QWORD *)(v5 + 8));
          return v5;
        }
        return v4;
      }
      if ( *(_BYTE *)v4 == 54 && *(_QWORD *)(v4 - 64) )
      {
        v14 = *(unsigned __int8 **)(v4 - 32);
        v60 = *(_QWORD *)(v4 - 64);
        v11 = *v14;
        if ( (_BYTE)v11 == 17 )
        {
          v59 = v14 + 24;
          goto LABEL_45;
        }
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v14 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v11 <= 0x15u )
        {
          v28 = sub_AD7630((__int64)v14, 0, v11);
          if ( v28 )
          {
            if ( *v28 == 17 )
            {
              v59 = v28 + 24;
LABEL_45:
              v15 = v58;
              sub_9865C0((__int64)&v70, (__int64)v58);
              sub_987160((__int64)&v70, (__int64)v15, v16, v17, v18);
              v19 = v71;
              v71 = 0;
              LODWORD(v73) = v19;
              v54 = (__int64)v59;
              v72 = v70;
              sub_9865C0((__int64)&v75, (__int64)&v72);
              sub_C48380((__int64)&v75, v54);
              LOBYTE(v54) = sub_9867B0((__int64)&v75);
              sub_969240((__int64 *)&v75);
              sub_969240((__int64 *)&v72);
              v9 = v4;
              sub_969240((__int64 *)&v70);
              if ( (_BYTE)v54 )
                return v9;
            }
          }
        }
      }
      v12 = *(_BYTE *)v4 == 55;
      LOBYTE(v77) = 0;
      v75 = &v60;
      v76 = (__int64 *)&v59;
      if ( v12 )
      {
        if ( *(_QWORD *)(v4 - 64) )
        {
          v23 = *(_QWORD *)(v4 - 32);
          v60 = *(_QWORD *)(v4 - 64);
          if ( (unsigned __int8)sub_991580((__int64)&v76, v23) )
          {
            v24 = v58;
            sub_9865C0((__int64)&v68, (__int64)v58);
            sub_987160((__int64)&v68, (__int64)v24, v25, v26, v27);
            v71 = v69;
            v56 = (__int64)v59;
            v70 = (__int64 *)v68;
            v69 = 0;
            sub_9865C0((__int64)&v72, (__int64)&v70);
            sub_C47AC0((__int64)&v72, v56);
            LOBYTE(v56) = sub_9867B0((__int64)&v72);
            sub_969240((__int64 *)&v72);
            sub_969240((__int64 *)&v70);
            v9 = v4;
            sub_969240((__int64 *)&v68);
            if ( (_BYTE)v56 )
              return v9;
          }
        }
      }
      goto LABEL_17;
    }
  }
  return sub_AD6530(*(_QWORD *)(v4 + 8), v5);
}
