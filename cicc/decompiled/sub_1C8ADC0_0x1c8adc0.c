// Function: sub_1C8ADC0
// Address: 0x1c8adc0
//
unsigned __int64 __fastcall sub_1C8ADC0(
        _BYTE *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 **v13; // rax
  __int64 ***v14; // r12
  char v15; // bl
  unsigned __int64 result; // rax
  char v17; // r13
  __int64 v18; // rax
  unsigned __int8 *v19; // rsi
  __int64 v20; // rax
  __int64 **v21; // rdx
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r11
  __int64 v26; // r13
  double v27; // xmm4_8
  double v28; // xmm5_8
  unsigned __int8 *v29; // rsi
  __int64 **v30; // rdx
  __int64 **v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r11
  unsigned __int64 *v41; // r13
  __int64 **v42; // rax
  unsigned __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // rdx
  unsigned __int8 *v46; // rsi
  _QWORD *v47; // [rsp+8h] [rbp-128h]
  __int64 **v48; // [rsp+10h] [rbp-120h]
  __int64 v49; // [rsp+18h] [rbp-118h]
  __int64 v52; // [rsp+30h] [rbp-100h]
  __int64 v53; // [rsp+30h] [rbp-100h]
  __int64 v54; // [rsp+30h] [rbp-100h]
  __int64 v55; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v57; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v58[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v59; // [rsp+60h] [rbp-D0h]
  __int64 v60[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v61; // [rsp+80h] [rbp-B0h]
  unsigned __int8 *v62[2]; // [rsp+90h] [rbp-A0h] BYREF
  _QWORD v63[2]; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int8 *v64; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v65; // [rsp+B8h] [rbp-78h]
  unsigned __int64 *v66; // [rsp+C0h] [rbp-70h]
  __int64 v67; // [rsp+C8h] [rbp-68h]
  __int64 v68; // [rsp+D0h] [rbp-60h]
  int v69; // [rsp+D8h] [rbp-58h]
  __int64 v70; // [rsp+E0h] [rbp-50h]
  __int64 v71; // [rsp+E8h] [rbp-48h]

  if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
    v13 = (unsigned __int64 **)*(a2 - 1);
  else
    v13 = (unsigned __int64 **)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
  v14 = (__int64 ***)*v13;
  v15 = *(_BYTE *)(*a2 + 8LL);
  result = **v13;
  v17 = *(_BYTE *)(result + 8);
  if ( v15 == 5 || v17 == 5 )
  {
    v47 = (_QWORD *)sub_16498A0((__int64)a2);
    v49 = *(_QWORD *)(*(_QWORD *)(a2[5] + 56LL) + 40LL);
    v18 = sub_16498A0((__int64)a2);
    v19 = (unsigned __int8 *)a2[6];
    v64 = 0;
    v67 = v18;
    v20 = a2[5];
    v68 = 0;
    v65 = v20;
    v66 = a2 + 3;
    v69 = 0;
    v70 = 0;
    v71 = 0;
    v62[0] = v19;
    if ( v19 )
    {
      sub_1623A60((__int64)v62, (__int64)v19, 2);
      if ( v64 )
        sub_161E7C0((__int64)&v64, (__int64)v64);
      v64 = v62[0];
      if ( v62[0] )
        sub_1623210((__int64)v62, v62[0], (__int64)&v64);
    }
    if ( v17 == 5 )
      v21 = (__int64 **)sub_1643370(v47);
    else
      v21 = *v14;
    if ( v15 == 5 )
    {
      v48 = v21;
      v33 = sub_1643370(v47);
      v21 = v48;
      v22 = (__int64 *)v33;
    }
    else
    {
      v22 = (__int64 *)*a2;
    }
    v63[0] = v21;
    v62[0] = (unsigned __int8 *)v63;
    v62[1] = (unsigned __int8 *)0x100000001LL;
    v23 = sub_1644EA0(v22, v63, 1, 0);
    v24 = sub_1632080(v49, a3, a4, v23, 0);
    v25 = v24;
    if ( (_QWORD *)v62[0] != v63 )
    {
      v52 = v24;
      _libc_free((unsigned __int64)v62[0]);
      v25 = v52;
    }
    v61 = 257;
    if ( v17 == 5 )
    {
      v53 = v25;
      v59 = 257;
      v31 = (__int64 **)sub_1643370(v47);
      v25 = v53;
      if ( v31 != *v14 )
      {
        if ( *((_BYTE *)v14 + 16) > 0x10u )
        {
          LOWORD(v63[0]) = 257;
          v39 = sub_15FDBD0(47, (__int64)v14, (__int64)v31, (__int64)v62, 0);
          v40 = v53;
          v14 = (__int64 ***)v39;
          if ( v65 )
          {
            v41 = v66;
            sub_157E9D0(v65 + 40, v39);
            v42 = v14[3];
            v40 = v53;
            v43 = *v41;
            v14[4] = (__int64 **)v41;
            v43 &= 0xFFFFFFFFFFFFFFF8LL;
            v14[3] = (__int64 **)(v43 | (unsigned __int8)v42 & 7);
            *(_QWORD *)(v43 + 8) = v14 + 3;
            *v41 = *v41 & 7 | (unsigned __int64)(v14 + 3);
          }
          v54 = v40;
          sub_164B780((__int64)v14, v58);
          v25 = v54;
          if ( v64 )
          {
            v57 = v64;
            sub_1623A60((__int64)&v57, (__int64)v64, 2);
            v44 = (__int64)v14[6];
            v45 = (__int64)(v14 + 6);
            v25 = v54;
            if ( v44 )
            {
              sub_161E7C0((__int64)(v14 + 6), v44);
              v25 = v54;
              v45 = (__int64)(v14 + 6);
            }
            v46 = v57;
            v14[6] = (__int64 **)v57;
            if ( v46 )
            {
              v55 = v25;
              sub_1623210((__int64)&v57, v46, v45);
              v25 = v55;
            }
          }
        }
        else
        {
          v32 = sub_15A46C0(47, v14, v31, 0);
          v25 = v53;
          v14 = (__int64 ***)v32;
        }
      }
    }
    v62[0] = (unsigned __int8 *)v14;
    v26 = sub_1285290((__int64 *)&v64, *(_QWORD *)(*(_QWORD *)v25 + 24LL), v25, (int)v62, 1, (__int64)v60, 0);
    if ( v15 == 5 )
    {
      v61 = 257;
      v30 = (__int64 **)sub_16432F0(v47);
      if ( v30 != *(__int64 ***)v26 )
      {
        if ( *(_BYTE *)(v26 + 16) > 0x10u )
        {
          LOWORD(v63[0]) = 257;
          v26 = sub_15FDBD0(47, v26, (__int64)v30, (__int64)v62, 0);
          if ( v65 )
          {
            v34 = (__int64 *)v66;
            sub_157E9D0(v65 + 40, v26);
            v35 = *(_QWORD *)(v26 + 24);
            v36 = *v34;
            *(_QWORD *)(v26 + 32) = v34;
            v36 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v26 + 24) = v36 | v35 & 7;
            *(_QWORD *)(v36 + 8) = v26 + 24;
            *v34 = *v34 & 7 | (v26 + 24);
          }
          sub_164B780(v26, v60);
          if ( v64 )
          {
            v58[0] = (__int64)v64;
            sub_1623A60((__int64)v58, (__int64)v64, 2);
            v37 = *(_QWORD *)(v26 + 48);
            if ( v37 )
              sub_161E7C0(v26 + 48, v37);
            v38 = (unsigned __int8 *)v58[0];
            *(_QWORD *)(v26 + 48) = v58[0];
            if ( v38 )
              sub_1623210((__int64)v58, v38, v26 + 48);
          }
        }
        else
        {
          v26 = sub_15A46C0(47, (__int64 ***)v26, v30, 0);
        }
      }
    }
    sub_164D160((__int64)a2, v26, a5, a6, a7, a8, v27, v28, a11, a12);
    sub_15F20C0(a2);
    result = (unsigned __int64)a1;
    v29 = v64;
    *a1 = 1;
    if ( v29 )
      return sub_161E7C0((__int64)&v64, (__int64)v29);
  }
  return result;
}
