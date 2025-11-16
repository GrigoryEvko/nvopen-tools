// Function: sub_1C8B580
// Address: 0x1c8b580
//
__int64 __fastcall sub_1C8B580(
        _BYTE *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 result; // rax
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r12
  __int16 **v15; // r13
  __int16 *v16; // rax
  __int16 *v17; // rax
  int v18; // eax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 **v21; // rsi
  char v22; // al
  char *v23; // rax
  char v24; // al
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 *v28; // rdi
  float v29; // xmm0_4
  __int64 *v30; // rdi
  float v31; // xmm0_4
  __int16 **v32; // r12
  __int16 *v33; // rax
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // r12
  __int64 v40; // r14
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // r12
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  __int64 v48; // rbx
  __int64 v49; // r12
  __int64 v50; // r15
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // r13
  __int64 v54; // rbx
  __int64 v55; // r12
  __int64 v56; // [rsp+0h] [rbp-100h]
  __int16 **v57; // [rsp+0h] [rbp-100h]
  __int64 v58; // [rsp+8h] [rbp-F8h]
  __int16 **v59; // [rsp+8h] [rbp-F8h]
  __int16 **v60; // [rsp+10h] [rbp-F0h]
  __int64 v61; // [rsp+10h] [rbp-F0h]
  __int16 *v62; // [rsp+18h] [rbp-E8h]
  __int64 v63; // [rsp+18h] [rbp-E8h]
  __int64 v64; // [rsp+18h] [rbp-E8h]
  __int64 v66; // [rsp+28h] [rbp-D8h]
  __int64 v67; // [rsp+30h] [rbp-D0h]
  __int64 v68; // [rsp+40h] [rbp-C0h]
  float v69; // [rsp+40h] [rbp-C0h]
  float v70; // [rsp+40h] [rbp-C0h]
  __int64 v71; // [rsp+40h] [rbp-C0h]
  __int16 *v72; // [rsp+40h] [rbp-C0h]
  __int64 v73; // [rsp+48h] [rbp-B8h]
  __int16 *v74; // [rsp+50h] [rbp-B0h]
  __int64 v75; // [rsp+58h] [rbp-A8h]
  char v76; // [rsp+6Fh] [rbp-91h] BYREF
  __int64 v77[4]; // [rsp+70h] [rbp-90h] BYREF
  char v78[8]; // [rsp+90h] [rbp-70h] BYREF
  __int16 *v79; // [rsp+98h] [rbp-68h] BYREF
  __int64 v80; // [rsp+A0h] [rbp-60h]
  __int16 *v81; // [rsp+B8h] [rbp-48h] BYREF
  __int64 v82; // [rsp+C0h] [rbp-40h]

  result = *(_QWORD *)(a2 + 32);
  v66 = a2 + 24;
  v67 = result;
  if ( a2 + 24 != result )
  {
    while ( 1 )
    {
      result = *(_QWORD *)(v67 + 24);
      v73 = v67 + 16;
      v67 = *(_QWORD *)(v67 + 8);
      v75 = result;
      if ( v73 != result )
        break;
LABEL_25:
      if ( v66 == v67 )
        return result;
    }
    while ( 1 )
    {
      v11 = *(_QWORD *)(v75 + 24);
      v12 = v75 + 16;
      v75 = *(_QWORD *)(v75 + 8);
      if ( v12 != v11 )
        break;
LABEL_24:
      if ( v73 == v75 )
        goto LABEL_25;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = v11;
        v11 = *(_QWORD *)(v11 + 8);
        if ( *(_BYTE *)(v13 - 8) == 43 )
        {
          result = *(_QWORD *)(v13 - 72);
          if ( !result )
            goto LABEL_27;
          if ( *(_BYTE *)(result + 16) > 0x10u )
          {
            v14 = *(_QWORD *)(v13 - 48);
            if ( !v14 )
LABEL_27:
              BUG();
            if ( *(_BYTE *)(v14 + 16) == 14 )
              break;
          }
        }
LABEL_5:
        if ( v12 == v11 )
          goto LABEL_24;
      }
      v15 = &v79;
      v68 = *(_QWORD *)(v14 + 32);
      v16 = (__int16 *)sub_16982C0();
      v74 = v16;
      if ( (__int16 *)v68 == v16 )
        sub_169C630(&v79, (__int64)v16, 1);
      else
        sub_1699170((__int64)&v79, v68, 1);
      if ( v79 == v74 )
        break;
      if ( !(unsigned __int8)sub_169DE70((__int64)v78) && !(unsigned __int8)sub_169DE70(v14 + 24) )
      {
        v17 = (__int16 *)sub_1698270();
        if ( v79 != v17
          || ((v62 = v17, (v23 = (char *)sub_16D40F0((__int64)qword_4FBB490)) == 0)
            ? (v24 = qword_4FBB490[2])
            : (v24 = *v23),
              !v24) )
        {
          v18 = sub_16994B0(&v79, v14 + 32, 0) & 0xFFFFFFEF;
          goto LABEL_18;
        }
        if ( !(unsigned __int8)sub_169DE70((__int64)v78) && !(unsigned __int8)sub_169DE70(v14 + 24) )
        {
          v28 = (__int64 *)(v14 + 32);
          if ( v74 == *(__int16 **)(v14 + 32) )
            v28 = (__int64 *)(*(_QWORD *)(v14 + 40) + 8LL);
          v29 = sub_169D890(v28);
          v30 = (__int64 *)&v79;
          if ( v74 == v79 )
            v30 = (__int64 *)(v80 + 8);
          v69 = v29;
          v31 = sub_169D890(v30);
          *(_QWORD *)&a4 = LODWORD(v69);
          v70 = sub_1C40E30(v31, v69, (__int64)&v76, 1, 1);
          a3 = (__m128)LODWORD(v70);
          if ( !sub_1C40EE0(v70) )
          {
            v32 = &v81;
            sub_169D3B0((__int64)v77, (__m128i)LODWORD(v70));
            sub_169E320(&v81, v77, v62);
            sub_1698460((__int64)v77);
            if ( v74 == v79 )
            {
              v72 = v81;
              v45 = v80;
              if ( v74 != v81 )
              {
                if ( v80 )
                {
                  v46 = 32LL * *(_QWORD *)(v80 - 8);
                  if ( v80 + v46 != v80 )
                  {
                    v47 = v13;
                    v48 = v80 + v46;
                    v49 = v80;
                    do
                    {
                      v48 -= 32;
                      sub_127D120((_QWORD *)(v48 + 8));
                    }
                    while ( v48 != v49 );
                    v45 = v49;
                    v13 = v47;
                    v32 = &v81;
                    v15 = &v79;
                  }
                  j_j_j___libc_free_0_0(v45 - 8);
                  v33 = v81;
                  goto LABEL_55;
                }
LABEL_56:
                sub_1698450((__int64)&v79, (__int64)&v81);
                goto LABEL_52;
              }
              if ( v80 )
              {
                if ( v80 + 32LL * *(_QWORD *)(v80 - 8) != v80 )
                {
                  v64 = v12;
                  v50 = v80;
                  v61 = v13;
                  v51 = v80 + 32LL * *(_QWORD *)(v80 - 8);
                  do
                  {
                    v51 -= 32;
                    if ( v72 == *(__int16 **)(v51 + 8) )
                    {
                      v52 = *(_QWORD *)(v51 + 16);
                      if ( v52 )
                      {
                        if ( v52 != v52 + 32LL * *(_QWORD *)(v52 - 8) )
                        {
                          v59 = v15;
                          v53 = v51;
                          v54 = v52 + 32LL * *(_QWORD *)(v52 - 8);
                          v57 = v32;
                          v55 = v52;
                          do
                          {
                            v54 -= 32;
                            sub_127D120((_QWORD *)(v54 + 8));
                          }
                          while ( v55 != v54 );
                          v52 = v55;
                          v51 = v53;
                          v32 = v57;
                          v15 = v59;
                        }
                        j_j_j___libc_free_0_0(v52 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v51 + 8);
                    }
                  }
                  while ( v51 != v50 );
                  v45 = v50;
                  v13 = v61;
                  v12 = v64;
                }
                j_j_j___libc_free_0_0(v45 - 8);
              }
            }
            else
            {
              if ( v74 != v81 )
              {
                sub_16983E0((__int64)&v79, (__int64)&v81);
LABEL_52:
                if ( v74 == v81 )
                {
                  v34 = v82;
                  if ( v82 )
                  {
                    v35 = 32LL * *(_QWORD *)(v82 - 8);
                    if ( v82 != v82 + v35 )
                    {
                      v63 = v13;
                      v36 = v82 + v35;
                      v71 = v11;
                      v60 = v15;
                      do
                      {
                        v36 -= 32;
                        if ( v74 == *(__int16 **)(v36 + 8) )
                        {
                          v37 = *(_QWORD *)(v36 + 16);
                          if ( v37 )
                          {
                            v38 = 32LL * *(_QWORD *)(v37 - 8);
                            if ( v37 != v37 + v38 )
                            {
                              v58 = v34;
                              v39 = v37 + v38;
                              v40 = v36;
                              do
                              {
                                v39 -= 32;
                                if ( v74 == *(__int16 **)(v39 + 8) )
                                {
                                  v41 = *(_QWORD *)(v39 + 16);
                                  if ( v41 )
                                  {
                                    v42 = 32LL * *(_QWORD *)(v41 - 8);
                                    if ( v41 != v41 + v42 )
                                    {
                                      v56 = v39;
                                      v43 = *(_QWORD *)(v39 + 16);
                                      v44 = v41 + v42;
                                      do
                                      {
                                        v44 -= 32;
                                        sub_127D120((_QWORD *)(v44 + 8));
                                      }
                                      while ( v43 != v44 );
                                      v41 = v43;
                                      v39 = v56;
                                    }
                                    j_j_j___libc_free_0_0(v41 - 8);
                                  }
                                }
                                else
                                {
                                  sub_1698460(v39 + 8);
                                }
                              }
                              while ( v37 != v39 );
                              v34 = v58;
                              v36 = v40;
                            }
                            j_j_j___libc_free_0_0(v37 - 8);
                          }
                        }
                        else
                        {
                          sub_1698460(v36 + 8);
                        }
                      }
                      while ( v34 != v36 );
                      v11 = v71;
                      v13 = v63;
                      v15 = v60;
                    }
                    j_j_j___libc_free_0_0(v34 - 8);
                  }
                }
                else
                {
                  sub_1698460((__int64)v32);
                }
LABEL_19:
                v21 = (__int64 **)(v13 - 24);
                v22 = *(_BYTE *)(*(_QWORD *)(v13 - 24) + 8LL);
                if ( v22 == 3 )
                {
                  sub_1C8B360(
                    a1,
                    v21,
                    (__int64)"__nv_fdiv_by_const_dp",
                    21,
                    (__int64)v78,
                    a3,
                    a4,
                    a5,
                    a6,
                    v19,
                    v20,
                    a9,
                    a10);
                }
                else if ( v22 == 2 )
                {
                  sub_1C8B360(
                    a1,
                    v21,
                    (__int64)"__nv_fdiv_by_const_sp",
                    21,
                    (__int64)v78,
                    a3,
                    a4,
                    a5,
                    a6,
                    v19,
                    v20,
                    a9,
                    a10);
                }
                goto LABEL_22;
              }
              sub_1698460((__int64)&v79);
              v33 = v81;
LABEL_55:
              if ( v74 != v33 )
                goto LABEL_56;
            }
            sub_169C7E0(v15, v32);
            goto LABEL_52;
          }
        }
      }
      if ( v74 != v79 )
      {
        sub_16986F0(&v79, 0, 0, 0);
        result = (__int64)v74;
        if ( v74 != v79 )
          goto LABEL_23;
LABEL_35:
        v25 = v80;
        if ( v80 )
        {
          v26 = 32LL * *(_QWORD *)(v80 - 8);
          v27 = v80 + v26;
          if ( v80 != v80 + v26 )
          {
            do
            {
              v27 -= 32;
              sub_127D120((_QWORD *)(v27 + 8));
            }
            while ( v25 != v27 );
          }
          result = j_j_j___libc_free_0_0(v25 - 8);
        }
        goto LABEL_5;
      }
      sub_169CAA0((__int64)&v79, 0, 0, 0, a3.m128_f32[0]);
LABEL_22:
      result = (__int64)v74;
      if ( v74 == v79 )
        goto LABEL_35;
LABEL_23:
      result = sub_1698460((__int64)v15);
      if ( v12 == v11 )
        goto LABEL_24;
    }
    v18 = sub_169EEB0(&v79, v14 + 32, 0, a3.m128_f32[0]) & 0xFFFFFFEF;
LABEL_18:
    if ( v18 )
      goto LABEL_22;
    goto LABEL_19;
  }
  return result;
}
