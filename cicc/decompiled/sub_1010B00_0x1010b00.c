// Function: sub_1010B00
// Address: 0x1010b00
//
__int64 *__fastcall sub_1010B00(__int64 *a1, __int64 *a2, __m128i *a3, unsigned int a4)
{
  __int64 *v4; // r14
  __int64 v5; // r13
  unsigned __int8 v8; // al
  __int64 v9; // r15
  bool v11; // zf
  unsigned int v12; // eax
  __int64 v13; // r8
  unsigned int v14; // ecx
  _QWORD *v15; // rax
  unsigned __int16 v16; // ax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  int v26; // eax
  bool v27; // al
  __int64 v28; // rax
  unsigned __int16 v29; // ax
  _QWORD *v30; // [rsp+8h] [rbp-108h]
  __int64 v31; // [rsp+10h] [rbp-100h]
  bool v32; // [rsp+10h] [rbp-100h]
  unsigned __int64 v33; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v34; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int8 *v35; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD *v36; // [rsp+50h] [rbp-C0h] BYREF
  _QWORD *v37; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v38; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+68h] [rbp-A8h] BYREF
  _QWORD *v40; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int8 **v41; // [rsp+78h] [rbp-98h]
  _QWORD *v42; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int8 **v43; // [rsp+88h] [rbp-88h]
  _QWORD *v44; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v45; // [rsp+98h] [rbp-78h]
  unsigned __int8 *v46; // [rsp+A0h] [rbp-70h]
  __int64 v47; // [rsp+B0h] [rbp-60h] BYREF
  __int64 *v48; // [rsp+B8h] [rbp-58h]
  __int64 v49; // [rsp+C0h] [rbp-50h]
  unsigned __int8 **v50; // [rsp+C8h] [rbp-48h]
  int v51; // [rsp+D0h] [rbp-40h]
  unsigned __int8 **v52; // [rsp+D8h] [rbp-38h]

  v4 = a1;
  v5 = (__int64)a2;
  v8 = *(_BYTE *)a2;
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    if ( v8 > 0x15u )
    {
      v8 = *(_BYTE *)a1;
      v4 = a2;
      v5 = (__int64)a1;
    }
    else
    {
      v9 = sub_96E6C0(0x1Du, (__int64)a1, a2, a3->m128i_i64[0]);
      if ( v9 )
        return (__int64 *)v9;
      v8 = *(_BYTE *)a2;
    }
  }
  if ( v8 == 13 )
    return (__int64 *)v5;
  if ( !(unsigned __int8)sub_1003090((__int64)a3, (unsigned __int8 *)v5) )
  {
    v47 = 0;
    if ( !(unsigned __int8)sub_995B10((_QWORD **)&v47, v5) )
    {
      v9 = v5;
      if ( (__int64 *)v5 == v4 )
        return (__int64 *)v9;
      v9 = (__int64)v4;
      if ( (unsigned __int8)sub_FFFE90(v5) )
        return (__int64 *)v9;
      v9 = (__int64)sub_100B250(v4, (_BYTE *)v5);
      if ( v9 )
        return (__int64 *)v9;
      v9 = (__int64)sub_100B250((__int64 *)v5, v4);
      if ( v9 )
        return (__int64 *)v9;
      v9 = sub_FFEF50((char *)v4, (_BYTE *)v5, 29);
      if ( v9 )
        return (__int64 *)v9;
      v40 = 0;
      v41 = &v34;
      if ( !(unsigned __int8)sub_100AC40(&v40, 25, (unsigned __int8 *)v4)
        || (v43 = &v35, v42 = 0, !(unsigned __int8)sub_100AC90(&v42, 26, (unsigned __int8 *)v5)) )
      {
        v44 = 0;
        v45 = (__int64 *)&v34;
        if ( !(unsigned __int8)sub_100AC40(&v44, 25, (unsigned __int8 *)v5) )
          goto LABEL_16;
        v48 = (__int64 *)&v35;
        v47 = 0;
        if ( !(unsigned __int8)sub_100AC90((_QWORD **)&v47, 26, (unsigned __int8 *)v4) )
          goto LABEL_16;
      }
      v44 = &v42;
      LOBYTE(v45) = 0;
      v46 = v35;
      if ( !sub_100ACE0((__int64)&v44, 15, v34) )
      {
        v47 = (__int64)&v42;
        LOBYTE(v48) = 0;
        v49 = (__int64)v34;
        if ( !sub_100ACE0((__int64)&v47, 15, v35) )
          goto LABEL_16;
      }
      v30 = v42;
      v31 = *((_QWORD *)v34 + 1);
      v12 = sub_BCB060(v31);
      v13 = v31;
      v14 = *((_DWORD *)v30 + 2);
      v33 = v12;
      if ( v14 > 0x40 )
      {
        if ( v14 - (unsigned int)sub_C444A0((__int64)v30) > 0x40 )
        {
LABEL_16:
          LODWORD(v47) = 180;
          LODWORD(v48) = 0;
          v49 = (__int64)&v34;
          LODWORD(v50) = 1;
          v51 = 2;
          v52 = &v35;
          if ( !(unsigned __int8)sub_1007380((__int64)&v47, (__int64)v4)
            || (v44 = v34, v45 = (__int64 *)v35, !sub_100AD70(&v44, 25, (unsigned __int8 *)v5)) )
          {
            LODWORD(v47) = 180;
            LODWORD(v48) = 0;
            v49 = (__int64)&v34;
            LODWORD(v50) = 1;
            v51 = 2;
            v52 = &v35;
            if ( (unsigned __int8)sub_1007380((__int64)&v47, v5) )
            {
              v44 = v34;
              v45 = (__int64 *)v35;
              if ( sub_100AD70(&v44, 25, (unsigned __int8 *)v4) )
                return (__int64 *)v5;
            }
            v47 = 181;
            v49 = 1;
            v50 = &v34;
            v51 = 2;
            v52 = &v35;
            if ( !(unsigned __int8)sub_1007400((__int64)&v47, (__int64)v4)
              || (v44 = v34, v45 = (__int64 *)v35, !sub_100ADA0(&v44, 26, (unsigned __int8 *)v5)) )
            {
              v47 = 181;
              v49 = 1;
              v50 = &v34;
              v51 = 2;
              v52 = &v35;
              if ( (unsigned __int8)sub_1007400((__int64)&v47, v5) )
              {
                v44 = v34;
                v45 = (__int64 *)v35;
                if ( sub_100ADA0(&v44, 26, (unsigned __int8 *)v4) )
                  return (__int64 *)v5;
              }
              v9 = sub_1010290(0x1Du, v4, (unsigned __int8 *)v5, a3, a4);
              if ( v9 )
                return (__int64 *)v9;
              v9 = sub_1010290(0x1Du, (_BYTE *)v5, (unsigned __int8 *)v4, a3, a4);
              if ( v9 )
                return (__int64 *)v9;
              v9 = sub_1006BA0(a3, (char *)v4, (char *)v5, 0);
              if ( v9 )
                return (__int64 *)v9;
              v9 = v5;
              if ( (unsigned __int8)sub_104A6F0(v4, v5, 0) )
                return (__int64 *)v9;
              v9 = (__int64)v4;
              if ( (unsigned __int8)sub_104A6F0(v5, v4, 0) )
                return (__int64 *)v9;
              v9 = sub_101B370(29, v4, v5, a3, a4);
              if ( v9 )
                return (__int64 *)v9;
              v9 = sub_101C7F0(29, v4, v5, 28, a3, a4);
              if ( v9 )
                return (__int64 *)v9;
              if ( *(_BYTE *)v4 != 86 && *(_BYTE *)v5 != 86 )
                goto LABEL_33;
              if ( !sub_1001970(v4[1], 1) )
                goto LABEL_88;
              v11 = *(_BYTE *)v5 == 86;
              v47 = (__int64)v4;
              v48 = 0;
              if ( v11 && (unsigned __int8)sub_FFF5F0(&v47, v5) )
                return (__int64 *)v5;
              v11 = *(_BYTE *)v4 == 86;
              v47 = v5;
              v48 = 0;
              if ( !v11 || !(unsigned __int8)sub_FFF5F0(&v47, (__int64)v4) )
              {
LABEL_88:
                v28 = sub_101C8A0(29, v4, v5, a3, a4);
                if ( v28 )
                  return (__int64 *)v28;
LABEL_33:
                v11 = *(_BYTE *)v4 == 57;
                LOBYTE(v46) = 0;
                v44 = &v36;
                v45 = &v38;
                if ( v11 )
                {
                  if ( (unsigned __int8)sub_FFF3B0((__int64)&v44, (__int64)v4) )
                  {
                    v11 = *(_BYTE *)v5 == 57;
                    LOBYTE(v49) = 0;
                    v47 = (__int64)&v37;
                    v48 = &v39;
                    if ( v11 )
                    {
                      if ( (unsigned __int8)sub_FFF3B0((__int64)&v47, v5) )
                      {
                        v22 = v39;
                        sub_9865C0((__int64)&v44, v39);
                        sub_987160((__int64)&v44, v22, v23, v24, v25);
                        v26 = (int)v45;
                        LODWORD(v45) = 0;
                        LODWORD(v48) = v26;
                        v47 = (__int64)v44;
                        v27 = *(_DWORD *)(v38 + 8) <= 0x40u
                            ? *(_QWORD *)v38 == (_QWORD)v44
                            : sub_C43C50(v38, (const void **)&v47);
                        v32 = v27;
                        sub_969240(&v47);
                        sub_969240((__int64 *)&v44);
                        if ( v32 )
                        {
                          if ( sub_1002450(v39) )
                          {
                            v47 = (__int64)v37;
                            v48 = (__int64 *)&v44;
                            if ( *(_BYTE *)v36 == 42
                              && (unsigned __int8)sub_FFE680((__int64)&v47, (__int64)v36)
                              && (unsigned __int8)sub_9AC230((__int64)v44, v39, a3, 0) )
                            {
                              return v36;
                            }
                          }
                          if ( sub_1002450(v38) )
                          {
                            v47 = (__int64)v36;
                            v48 = (__int64 *)&v44;
                            if ( *(_BYTE *)v37 == 42
                              && (unsigned __int8)sub_FFE680((__int64)&v47, (__int64)v37)
                              && (unsigned __int8)sub_9AC230((__int64)v44, v38, a3, 0) )
                            {
                              return v37;
                            }
                          }
                        }
                      }
                    }
                  }
                }
                if ( *(_BYTE *)v4 == 84 || *(_BYTE *)v5 == 84 )
                {
                  v28 = sub_101CAB0(29, v4, v5, a3, a4);
                  if ( v28 )
                    return (__int64 *)v28;
                }
                v11 = *(_BYTE *)v4 == 59;
                LOBYTE(v49) = 0;
                v47 = (__int64)&v36;
                v48 = &v38;
                if ( v11 && (unsigned __int8)sub_FFF310((__int64)&v47, (__int64)v4) )
                {
                  v17 = v38;
                  sub_9865C0((__int64)&v40, v38);
                  sub_987160((__int64)&v40, v17, v18, v19, v20);
                  v21 = (int)v41;
                  LODWORD(v41) = 0;
                  LODWORD(v43) = v21;
                  v45 = (__int64 *)&v42;
                  v42 = v40;
                  v44 = v36;
                  if ( sub_100AAA0((__int64)&v44, 30, (unsigned __int8 *)v5) )
                  {
                    sub_969240((__int64 *)&v42);
                    sub_969240((__int64 *)&v40);
                    return (__int64 *)sub_AD62B0(v4[1]);
                  }
                  sub_969240((__int64 *)&v42);
                  sub_969240((__int64 *)&v40);
                }
                if ( !sub_1001970(v4[1], 1) )
                {
LABEL_38:
                  if ( a4 == 3 )
                    return (__int64 *)sub_FFE7D0(
                                        29,
                                        (__int64)v4,
                                        (unsigned __int8 *)v5,
                                        a3->m128i_i64[0],
                                        a3[2].m128i_i64[1]);
                  return (__int64 *)v9;
                }
                v16 = sub_9A18B0((__int64)v4, (_BYTE *)v5, a3->m128i_i64[0], 0, 0);
                if ( HIBYTE(v16) )
                {
                  if ( (_BYTE)v16 )
                    return (__int64 *)sub_AD6400(v4[1]);
                  return v4;
                }
                v29 = sub_9A18B0(v5, v4, a3->m128i_i64[0], 0, 0);
                if ( !HIBYTE(v29) )
                  goto LABEL_38;
                if ( (_BYTE)v29 )
                  return (__int64 *)sub_AD6400(*(_QWORD *)(v5 + 8));
                return (__int64 *)v5;
              }
            }
          }
          return v4;
        }
        v13 = v31;
        v15 = *(_QWORD **)*v30;
      }
      else
      {
        v15 = (_QWORD *)*v30;
      }
      if ( v33 >= (unsigned __int64)v15 )
        return (__int64 *)sub_AD62B0(v13);
      goto LABEL_16;
    }
  }
  return (__int64 *)sub_AD62B0(v4[1]);
}
