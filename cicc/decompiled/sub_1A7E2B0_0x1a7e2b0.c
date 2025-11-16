// Function: sub_1A7E2B0
// Address: 0x1a7e2b0
//
__int64 __fastcall sub_1A7E2B0(
        _QWORD *a1,
        __m128 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        _QWORD *a11,
        __int64 *a12)
{
  _DWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *v15; // r14
  _QWORD *v16; // r13
  _QWORD *i; // rbx
  _QWORD *v18; // r12
  unsigned __int8 v19; // al
  _QWORD *v20; // rax
  __int64 v21; // r13
  unsigned __int64 v22; // rax
  __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  double v25; // xmm4_8
  double v26; // xmm5_8
  _QWORD *v27; // rax
  _BYTE *v28; // r13
  _BYTE *v29; // rbx
  _QWORD *v30; // r12
  __int64 v31; // r8
  __int64 v32; // rsi
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // r15
  _QWORD *v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r15
  unsigned __int64 v39; // rax
  int v40; // r8d
  int v41; // r9d
  _QWORD *v42; // rdi
  unsigned __int32 v43; // edx
  _QWORD *v44; // rdi
  __int64 v45; // r14
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  double v48; // xmm4_8
  double v49; // xmm5_8
  __int64 v50; // r11
  __int64 v51; // r14
  __int64 v52; // rbx
  __int64 v53; // [rsp+0h] [rbp-150h]
  unsigned __int64 v54; // [rsp+8h] [rbp-148h]
  __int64 v55; // [rsp+18h] [rbp-138h]
  char v56; // [rsp+18h] [rbp-138h]
  unsigned __int64 v57; // [rsp+18h] [rbp-138h]
  __int64 v58; // [rsp+18h] [rbp-138h]
  __int64 v59; // [rsp+18h] [rbp-138h]
  _QWORD *v60; // [rsp+30h] [rbp-120h]
  _QWORD *v62; // [rsp+40h] [rbp-110h]
  char v65; // [rsp+66h] [rbp-EAh]
  unsigned __int8 v66; // [rsp+67h] [rbp-E9h]
  _QWORD *v67; // [rsp+68h] [rbp-E8h]
  char v68; // [rsp+76h] [rbp-DAh] BYREF
  bool v69; // [rsp+77h] [rbp-D9h] BYREF
  __int64 v70; // [rsp+78h] [rbp-D8h] BYREF
  _BYTE *v71; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+88h] [rbp-C8h]
  _BYTE v73[64]; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v74; // [rsp+D0h] [rbp-80h] BYREF
  _QWORD v75[14]; // [rsp+E0h] [rbp-70h] BYREF

  v74.m128i_i64[0] = sub_1560340(a1 + 14, -1, "disable-tail-calls", 0x12u);
  v12 = (_DWORD *)sub_155D8B0(v74.m128i_i64);
  if ( v13 == 4 && *v12 == 1702195828 )
  {
    return 0;
  }
  else
  {
    v68 = 0;
    v66 = sub_1A7B710((__int64)a1, &v68, a12);
    v65 = v68;
    if ( v68 )
    {
      if ( *(_DWORD *)(a1[3] + 8LL) >> 8 )
      {
        return 0;
      }
      else
      {
        v15 = (_QWORD *)a1[10];
        v16 = a1 + 9;
        v70 = 0;
        v71 = v73;
        v69 = 0;
        v72 = 0x800000000LL;
        if ( a1 + 9 == v15 )
        {
          i = 0;
        }
        else
        {
          if ( !v15 )
            BUG();
          while ( 1 )
          {
            i = (_QWORD *)v15[3];
            if ( i != v15 + 2 )
              break;
            v15 = (_QWORD *)v15[1];
            if ( v16 == v15 )
              break;
            if ( !v15 )
              BUG();
          }
        }
        while ( v16 != v15 )
        {
          if ( !i )
            BUG();
          if ( *((_BYTE *)i - 8) == 53 && !(unsigned __int8)sub_15F8F00((__int64)(i - 3)) )
            break;
          for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v15[3] )
          {
            v27 = v15 - 3;
            if ( !v15 )
              v27 = 0;
            if ( i != v27 + 5 )
              break;
            v15 = (_QWORD *)v15[1];
            if ( v16 == v15 )
              goto LABEL_16;
            if ( !v15 )
              BUG();
          }
        }
LABEL_16:
        v18 = (_QWORD *)a1[10];
        if ( v18 != v16 )
        {
          v67 = a1 + 9;
          v60 = v15;
          do
          {
            v20 = v18;
            v62 = v18;
            v18 = (_QWORD *)v18[1];
            v21 = (__int64)(v20 - 3);
            v22 = sub_157EBA0((__int64)(v20 - 3));
            v23 = v22;
            if ( *(_BYTE *)(v22 + 16) == 25 )
            {
              v24 = sub_1A7A270((_QWORD *)v22, v67 != v60);
              if ( !v24
                || (v19 = sub_1A7CAB0(v24, v23, &v70, &v69, (__int64)&v71, a11, a2, a3, a4, a5, v25, v26, a8, a9, a12)) == 0 )
              {
                if ( v23 == sub_157ED60(v21) )
                {
                  v74.m128i_i64[0] = (__int64)v75;
                  v74.m128i_i64[1] = 0x800000000LL;
                  v35 = *(v62 - 2);
                  if ( v35 )
                  {
                    while ( 1 )
                    {
                      v36 = sub_1648700(v35);
                      if ( (unsigned __int8)(*((_BYTE *)v36 + 16) - 25) <= 9u )
                        break;
                      v35 = *(_QWORD *)(v35 + 8);
                      if ( !v35 )
                        goto LABEL_58;
                    }
                    v55 = v23;
                    v37 = v35;
                    v38 = 0;
LABEL_46:
                    v39 = sub_157EBA0(v36[5]);
                    if ( *(_BYTE *)(v39 + 16) == 26 && (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) == 1 )
                    {
                      if ( v74.m128i_i32[3] <= (unsigned int)v38 )
                      {
                        v54 = v39;
                        sub_16CD150((__int64)&v74, v75, 0, 8, v40, v41);
                        v38 = v74.m128i_u32[2];
                        v39 = v54;
                      }
                      *(_QWORD *)(v74.m128i_i64[0] + 8 * v38) = v39;
                      v38 = (unsigned int)++v74.m128i_i32[2];
                    }
                    while ( 1 )
                    {
                      v37 = *(_QWORD *)(v37 + 8);
                      if ( !v37 )
                        break;
                      v36 = sub_1648700(v37);
                      if ( (unsigned __int8)(*((_BYTE *)v36 + 16) - 25) <= 9u )
                        goto LABEL_46;
                    }
                    v23 = v55;
                    v42 = (_QWORD *)v74.m128i_i64[0];
                    v43 = v38;
                  }
                  else
                  {
LABEL_58:
                    v42 = v75;
                    v43 = 0;
                  }
                  v56 = 0;
                  while ( v43 )
                  {
                    v44 = (_QWORD *)v42[v43 - 1];
                    v74.m128i_i32[2] = v43 - 1;
                    v45 = v44[5];
                    v46 = sub_1A7A270(v44, v67 != v60);
                    if ( v46 )
                    {
                      v57 = v46;
                      v47 = sub_1AA6640(v23, v21, v45);
                      v50 = v57;
                      v51 = v47;
                      if ( !*((_WORD *)v62 - 3) )
                      {
                        if ( *(v62 - 2) )
                        {
                          v53 = v57;
                          v58 = v23;
                          v52 = *(v62 - 2);
                          do
                          {
                            if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v52) + 16) - 25) <= 9u )
                            {
                              v23 = v58;
                              v50 = v53;
                              goto LABEL_63;
                            }
                            v52 = *(_QWORD *)(v52 + 8);
                          }
                          while ( v52 );
                          v23 = v58;
                          v50 = v53;
                        }
                        v59 = v50;
                        sub_157F980(v21);
                        v50 = v59;
                      }
LABEL_63:
                      sub_1A7CAB0(v50, v51, &v70, &v69, (__int64)&v71, a11, a2, a3, a4, a5, v48, v49, a8, a9, a12);
                      v43 = v74.m128i_u32[2];
                      v42 = (_QWORD *)v74.m128i_i64[0];
                      v56 = v65;
                    }
                    else
                    {
                      v43 = v74.m128i_u32[2];
                      v42 = (_QWORD *)v74.m128i_i64[0];
                    }
                  }
                  if ( v42 != v75 )
                    _libc_free((unsigned __int64)v42);
                  v19 = v56 | v66;
                }
                else
                {
                  v19 = v66;
                }
              }
              v66 = v19;
            }
          }
          while ( v18 != v67 );
        }
        v28 = v71;
        v29 = &v71[8 * (unsigned int)v72];
        if ( v29 != v71 )
        {
          do
          {
            v30 = *(_QWORD **)v28;
            v74 = (__m128i)(unsigned __int64)sub_1632FA0(a1[5]);
            memset(v75, 0, 24);
            v32 = sub_13E3350((__int64)v30, &v74, 0, 1, v31);
            if ( v32 )
            {
              sub_164D160((__int64)v30, v32, a2, *(double *)a3.m128i_i64, a4, a5, v33, v34, a8, a9);
              sub_15F20C0(v30);
            }
            v28 += 8;
          }
          while ( v29 != v28 );
          v28 = v71;
        }
        if ( v28 != v73 )
          _libc_free((unsigned __int64)v28);
      }
    }
  }
  return v66;
}
