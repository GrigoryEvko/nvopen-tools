// Function: sub_21BF6F0
// Address: 0x21bf6f0
//
__int64 __fastcall sub_21BF6F0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // r13
  unsigned int v12; // r14d
  unsigned __int16 v13; // ax
  unsigned __int8 v14; // si
  int v15; // edx
  unsigned __int8 v16; // r15
  __int64 v18; // r9
  __int64 v19; // r11
  __int64 v20; // r10
  int v21; // edx
  int v22; // r15d
  __int64 v23; // rax
  _QWORD *v24; // rsi
  __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r9
  __int64 *v30; // r11
  __int64 v31; // r10
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // r9
  __int64 v42; // r11
  int v43; // edx
  __int64 v44; // r11
  __int16 v45; // ax
  __int64 v46; // rdi
  int v47; // edx
  __int64 v48; // [rsp+8h] [rbp-108h]
  __int64 v49; // [rsp+18h] [rbp-F8h]
  __int64 v50; // [rsp+18h] [rbp-F8h]
  __int64 v51; // [rsp+18h] [rbp-F8h]
  __int64 v52; // [rsp+20h] [rbp-F0h]
  __int64 v53; // [rsp+20h] [rbp-F0h]
  __int16 v54; // [rsp+2Ch] [rbp-E4h]
  __m128i v55; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v56; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v59; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v60; // [rsp+68h] [rbp-A8h] BYREF
  __int64 v61; // [rsp+70h] [rbp-A0h] BYREF
  int v62; // [rsp+78h] [rbp-98h]
  __int64 *v63; // [rsp+80h] [rbp-90h] BYREF
  __int64 v64; // [rsp+88h] [rbp-88h]
  char v65[8]; // [rsp+90h] [rbp-80h] BYREF
  __int64 v66; // [rsp+98h] [rbp-78h]
  unsigned __int8 v67; // [rsp+A0h] [rbp-70h]
  __int64 v68; // [rsp+A8h] [rbp-68h]
  unsigned __int8 v69; // [rsp+B0h] [rbp-60h]
  __int64 v70; // [rsp+B8h] [rbp-58h]
  char v71; // [rsp+C0h] [rbp-50h]
  __int64 v72; // [rsp+C8h] [rbp-48h]
  char v73; // [rsp+D0h] [rbp-40h]
  __int64 v74; // [rsp+D8h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = _mm_loadu_si128((const __m128i *)(v7 + 120));
  v61 = v8;
  v11 = *(_QWORD *)(v7 + 80);
  v56 = v9;
  v55 = v10;
  if ( v8 )
    sub_1623A60((__int64)&v61, v8, 2);
  v12 = 0;
  v62 = *(_DWORD *)(a2 + 64);
  v13 = *(_WORD *)(a2 + 24) - 667;
  if ( v13 <= 2u )
  {
    v14 = *(_BYTE *)(a2 + 88);
    v15 = dword_433D960[v13];
    v16 = **(_BYTE **)(a2 + 40);
    switch ( v15 )
    {
      case 2:
        v60 = 0x100000C28LL;
        v59 = 0x100000C26LL;
        v58 = 0x100000C25LL;
        v57 = 0x100000C2BLL;
        sub_21BD570(
          (__int64)&v63,
          v14,
          3116,
          3113,
          3114,
          (__int64)&v57,
          (__int64)&v58,
          (__int64)&v59,
          3111,
          (__int64)&v60);
        if ( BYTE4(v63) )
        {
          v54 = (__int16)v63;
          v20 = sub_1D238E0(*(_QWORD *)(a1 + 272), v16, v42, v16, v42, v41, 1, 0, 111, 0);
          v22 = v43;
          goto LABEL_12;
        }
        break;
      case 4:
        v59 = 0x100000C2ELL;
        BYTE4(v60) = 0;
        v58 = 0x100000C2DLL;
        BYTE4(v57) = 0;
        sub_21BD570(
          (__int64)&v63,
          v14,
          3122,
          3120,
          3121,
          (__int64)&v57,
          (__int64)&v58,
          (__int64)&v59,
          3119,
          (__int64)&v60);
        if ( BYTE4(v63) )
        {
          v45 = (__int16)v63;
          v46 = *(_QWORD *)(a1 + 272);
          v65[0] = v16;
          LOBYTE(v63) = v16;
          v67 = v16;
          v69 = v16;
          v54 = v45;
          v64 = v44;
          v66 = v44;
          v68 = v44;
          v70 = v44;
          v71 = 1;
          v72 = 0;
          v73 = 111;
          v74 = 0;
          v20 = sub_1D25C30(v46, (unsigned __int8 *)&v63, 6);
          v22 = v47;
          goto LABEL_12;
        }
        break;
      case 1:
        v60 = 0x100000C20LL;
        v59 = 0x100000C1ELL;
        v58 = 0x100000C1DLL;
        v57 = 0x100000C23LL;
        sub_21BD570(
          (__int64)&v63,
          v14,
          3108,
          3105,
          3106,
          (__int64)&v57,
          (__int64)&v58,
          (__int64)&v59,
          3103,
          (__int64)&v60);
        if ( BYTE4(v63) )
        {
          v54 = (__int16)v63;
          v20 = sub_1D25E70(*(_QWORD *)(a1 + 272), v16, v19, 1, 0, v18, 111, 0);
          v22 = v21;
LABEL_12:
          v23 = *(_QWORD *)(v11 + 88);
          v24 = *(_QWORD **)(v23 + 24);
          if ( *(_DWORD *)(v23 + 32) > 0x40u )
            v24 = (_QWORD *)*v24;
          v25 = *(_QWORD *)(a1 + 272);
          v49 = v20;
          v63 = (__int64 *)v65;
          v64 = 0x200000000LL;
          v27 = sub_1D38BB0(v25, (unsigned int)v24, (__int64)&v61, 5, 0, 1, v9, *(double *)v10.m128i_i64, a5, 0);
          v28 = (unsigned int)v64;
          v29 = v26;
          v30 = &v61;
          v31 = v49;
          if ( (unsigned int)v64 >= HIDWORD(v64) )
          {
            v48 = v49;
            v50 = v27;
            v52 = v26;
            sub_16CD150((__int64)&v63, v65, 0, 16, v27, v26);
            v28 = (unsigned int)v64;
            v31 = v48;
            v30 = &v61;
            v27 = v50;
            v29 = v52;
          }
          v32 = &v63[2 * v28];
          *v32 = v27;
          v32[1] = v29;
          v33 = (unsigned int)(v64 + 1);
          LODWORD(v64) = v33;
          if ( HIDWORD(v64) <= (unsigned int)v33 )
          {
            v51 = v31;
            sub_16CD150((__int64)&v63, v65, 0, 16, v27, v29);
            v33 = (unsigned int)v64;
            v31 = v51;
            v30 = &v61;
          }
          *(__m128i *)&v63[2 * v33] = _mm_load_si128(&v56);
          v34 = (unsigned int)(v64 + 1);
          LODWORD(v64) = v34;
          if ( HIDWORD(v64) <= (unsigned int)v34 )
          {
            v53 = v31;
            v56.m128i_i64[0] = (__int64)&v61;
            sub_16CD150((__int64)&v63, v65, 0, 16, v27, v29);
            v34 = (unsigned int)v64;
            v31 = v53;
            v30 = (__int64 *)v56.m128i_i64[0];
          }
          *(__m128i *)&v63[2 * v34] = _mm_load_si128(&v55);
          v35 = *(_QWORD **)(a1 + 272);
          LODWORD(v64) = v64 + 1;
          v36 = sub_1D23DE0(v35, v54, (__int64)v30, v31, v22, v29, v63, (unsigned int)v64);
          sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v36);
          sub_1D49010(v36);
          sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v37, v38, v39, v40);
          if ( v63 != (__int64 *)v65 )
            _libc_free((unsigned __int64)v63);
          v12 = 1;
          goto LABEL_7;
        }
        break;
      default:
        goto LABEL_7;
    }
    v12 = 0;
  }
LABEL_7:
  if ( v61 )
    sub_161E7C0((__int64)&v61, v61);
  return v12;
}
