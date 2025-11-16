// Function: sub_1EABF10
// Address: 0x1eabf10
//
__int64 __fastcall sub_1EABF10(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r14
  const char *v12; // rax
  unsigned __int64 v13; // rdx
  char v15; // r13
  _QWORD *v16; // rax
  __int64 *v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  _QWORD *v21; // rax
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int8 *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r11
  _QWORD *v27; // rax
  _QWORD *v28; // r14
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // rax
  __int64 v38; // r11
  __int64 *v39; // r14
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rdx
  unsigned __int8 *v44; // rsi
  __int64 v45; // [rsp+0h] [rbp-110h]
  unsigned __int8 i; // [rsp+17h] [rbp-F9h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  unsigned __int64 *v48; // [rsp+18h] [rbp-F8h]
  __int64 v49; // [rsp+18h] [rbp-F8h]
  __int64 v50; // [rsp+18h] [rbp-F8h]
  __int64 v51; // [rsp+18h] [rbp-F8h]
  __int64 v52; // [rsp+18h] [rbp-F8h]
  __int64 **v53; // [rsp+20h] [rbp-F0h]
  __int64 v54; // [rsp+28h] [rbp-E8h]
  __int64 v55; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v56; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v57[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v58; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v59[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v60; // [rsp+80h] [rbp-90h]
  unsigned __int8 *v61; // [rsp+90h] [rbp-80h] BYREF
  __int64 v62; // [rsp+98h] [rbp-78h]
  __int64 *v63; // [rsp+A0h] [rbp-70h]
  __int64 v64; // [rsp+A8h] [rbp-68h]
  __int64 v65; // [rsp+B0h] [rbp-60h]
  int v66; // [rsp+B8h] [rbp-58h]
  __int64 v67; // [rsp+C0h] [rbp-50h]
  __int64 v68; // [rsp+C8h] [rbp-48h]

  v9 = *(_QWORD *)(a1 + 32);
  for ( i = 0; a1 + 24 != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    v10 = 0;
    if ( v9 )
      v10 = v9 - 56;
    v11 = v10;
    v12 = sub_1649960(v10);
    if ( v13 > 0x12
      && !(*(_QWORD *)v12 ^ 0x616F6C2E6D766C6CLL | *((_QWORD *)v12 + 1) ^ 0x6974616C65722E64LL)
      && *((_WORD *)v12 + 8) == 25974
      && v12[18] == 46 )
    {
      if ( *(_QWORD *)(v11 + 8) )
      {
        v15 = 0;
        v16 = (_QWORD *)sub_15E0530(v11);
        v17 = (__int64 *)sub_1643350(v16);
        v53 = (__int64 **)sub_1647190(v17, 0);
        v18 = (_QWORD *)sub_15E0530(v11);
        v19 = sub_1643330(v18);
        v20 = *(_QWORD *)(v11 + 8);
        v54 = v19;
        if ( v20 )
        {
          v55 = v11;
          do
          {
            while ( 1 )
            {
              v21 = sub_1648700(v20);
              v20 = *(_QWORD *)(v20 + 8);
              v22 = (__int64)v21;
              if ( *((_BYTE *)v21 + 16) == 78 && v55 == *(v21 - 3) )
                break;
              if ( !v20 )
                goto LABEL_36;
            }
            v23 = sub_16498A0((__int64)v21);
            v67 = 0;
            v68 = 0;
            v24 = *(unsigned __int8 **)(v22 + 48);
            v64 = v23;
            v66 = 0;
            v25 = *(_QWORD *)(v22 + 40);
            v61 = 0;
            v62 = v25;
            v65 = 0;
            v63 = (__int64 *)(v22 + 24);
            v59[0] = v24;
            if ( v24 )
            {
              sub_1623A60((__int64)v59, (__int64)v24, 2);
              if ( v61 )
                sub_161E7C0((__int64)&v61, (__int64)v61);
              v61 = v59[0];
              if ( v59[0] )
                sub_1623210((__int64)v59, v59[0], (__int64)&v61);
            }
            v60 = 257;
            v26 = sub_12815B0(
                    (__int64 *)&v61,
                    v54,
                    *(_BYTE **)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF)),
                    *(_QWORD *)(v22 + 24 * (1LL - (*(_DWORD *)(v22 + 20) & 0xFFFFFFF))),
                    (__int64)v59);
            v58 = 257;
            if ( v53 != *(__int64 ***)v26 )
            {
              if ( *(_BYTE *)(v26 + 16) > 0x10u )
              {
                v60 = 257;
                v37 = sub_15FDBD0(47, v26, (__int64)v53, (__int64)v59, 0);
                v38 = v37;
                if ( v62 )
                {
                  v39 = v63;
                  v49 = v37;
                  sub_157E9D0(v62 + 40, v37);
                  v38 = v49;
                  v40 = *v39;
                  v41 = *(_QWORD *)(v49 + 24);
                  *(_QWORD *)(v49 + 32) = v39;
                  v40 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v49 + 24) = v40 | v41 & 7;
                  *(_QWORD *)(v40 + 8) = v49 + 24;
                  *v39 = *v39 & 7 | (v49 + 24);
                }
                v50 = v38;
                sub_164B780(v38, v57);
                v26 = v50;
                if ( v61 )
                {
                  v56 = v61;
                  sub_1623A60((__int64)&v56, (__int64)v61, 2);
                  v26 = v50;
                  v42 = *(_QWORD *)(v50 + 48);
                  v43 = v50 + 48;
                  if ( v42 )
                  {
                    v45 = v50;
                    v51 = v50 + 48;
                    sub_161E7C0(v51, v42);
                    v26 = v45;
                    v43 = v51;
                  }
                  v44 = v56;
                  *(_QWORD *)(v26 + 48) = v56;
                  if ( v44 )
                  {
                    v52 = v26;
                    sub_1623210((__int64)&v56, v44, v43);
                    v26 = v52;
                  }
                }
              }
              else
              {
                v26 = sub_15A46C0(47, (__int64 ***)v26, v53, 0);
              }
            }
            v47 = v26;
            v60 = 257;
            v27 = sub_1648A60(64, 1u);
            v28 = v27;
            if ( v27 )
              sub_15F9210((__int64)v27, *(_QWORD *)(*(_QWORD *)v47 + 24LL), v47, 0, 0, 0);
            if ( v62 )
            {
              v48 = (unsigned __int64 *)v63;
              sub_157E9D0(v62 + 40, (__int64)v28);
              v29 = *v48;
              v30 = v28[3] & 7LL;
              v28[4] = v48;
              v29 &= 0xFFFFFFFFFFFFFFF8LL;
              v28[3] = v29 | v30;
              *(_QWORD *)(v29 + 8) = v28 + 3;
              *v48 = *v48 & 7 | (unsigned __int64)(v28 + 3);
            }
            sub_164B780((__int64)v28, (__int64 *)v59);
            if ( v61 )
            {
              v57[0] = (__int64)v61;
              sub_1623A60((__int64)v57, (__int64)v61, 2);
              v31 = v28[6];
              v32 = (__int64)(v28 + 6);
              if ( v31 )
              {
                sub_161E7C0((__int64)(v28 + 6), v31);
                v32 = (__int64)(v28 + 6);
              }
              v33 = (unsigned __int8 *)v57[0];
              v28[6] = v57[0];
              if ( v33 )
                sub_1623210((__int64)v57, v33, v32);
            }
            sub_15F8F50((__int64)v28, 4u);
            v60 = 257;
            v34 = sub_12815B0(
                    (__int64 *)&v61,
                    v54,
                    *(_BYTE **)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF)),
                    (__int64)v28,
                    (__int64)v59);
            sub_164D160(v22, v34, a2, a3, a4, a5, v35, v36, a8, a9);
            sub_15F20C0((_QWORD *)v22);
            if ( v61 )
              sub_161E7C0((__int64)&v61, (__int64)v61);
            v15 = 1;
          }
          while ( v20 );
LABEL_36:
          i |= v15;
        }
      }
    }
  }
  return i;
}
