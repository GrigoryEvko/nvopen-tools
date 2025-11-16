// Function: sub_18B7320
// Address: 0x18b7320
//
__int64 __fastcall sub_18B7320(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        char a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // rbx
  unsigned __int64 v16; // r15
  __int64 v17; // r14
  _QWORD *v18; // rdi
  _DWORD *v19; // rax
  unsigned __int64 v20; // r15
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __int64 **v24; // rdx
  __int64 ***v25; // r14
  __int64 v26; // r9
  __int64 v27; // r13
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 ***v30; // rax
  __int64 **v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // r15
  __int64 **v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rsi
  __int64 *v37; // r14
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  __int64 *v43; // r15
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // rsi
  unsigned __int8 *v47; // rsi
  __int64 v48; // rsi
  __int64 v49; // rax
  unsigned __int64 *v50; // r15
  __int64 **v51; // rax
  unsigned __int64 v52; // rcx
  __int64 v53; // rsi
  unsigned __int8 *v54; // rsi
  __int64 result; // rax
  __int64 *v59; // [rsp+20h] [rbp-110h]
  __int16 v60; // [rsp+2Ch] [rbp-104h]
  __int64 v61; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v63; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v64[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v65; // [rsp+60h] [rbp-D0h]
  __int64 v66[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v67; // [rsp+80h] [rbp-B0h]
  unsigned __int8 *v68[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-90h]
  unsigned __int8 *v70; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-78h]
  unsigned __int64 *v72; // [rsp+C0h] [rbp-70h]
  __int64 v73; // [rsp+C8h] [rbp-68h]
  __int64 v74; // [rsp+D0h] [rbp-60h]
  int v75; // [rsp+D8h] [rbp-58h]
  __int64 v76; // [rsp+E0h] [rbp-50h]
  __int64 v77; // [rsp+E8h] [rbp-48h]

  v14 = *(_QWORD *)a2;
  v61 = *(_QWORD *)(a2 + 8);
  if ( *(_QWORD *)a2 != v61 )
  {
    v60 = (a5 == 0) + 32;
    do
    {
      v20 = *(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      v21 = sub_16498A0(v20);
      v76 = 0;
      v77 = 0;
      v22 = *(unsigned __int8 **)(v20 + 48);
      v73 = v21;
      v75 = 0;
      v23 = *(_QWORD *)(v20 + 40);
      v70 = 0;
      v71 = v23;
      v74 = 0;
      v72 = (unsigned __int64 *)(v20 + 24);
      v68[0] = v22;
      if ( v22 )
      {
        sub_1623A60((__int64)v68, (__int64)v22, 2);
        if ( v70 )
          sub_161E7C0((__int64)&v70, (__int64)v70);
        v70 = v68[0];
        if ( v68[0] )
          sub_1623210((__int64)v68, v68[0], (__int64)&v70);
      }
      v65 = 257;
      v24 = *(__int64 ***)(a1 + 48);
      v67 = 257;
      v25 = *(__int64 ****)v14;
      if ( v24 != **(__int64 ****)v14 )
      {
        if ( *((_BYTE *)v25 + 16) > 0x10u )
        {
          v48 = *(_QWORD *)v14;
          v69 = 257;
          v49 = sub_15FDBD0(47, v48, (__int64)v24, (__int64)v68, 0);
          v25 = (__int64 ***)v49;
          if ( v71 )
          {
            v50 = v72;
            sub_157E9D0(v71 + 40, v49);
            v51 = v25[3];
            v52 = *v50;
            v25[4] = (__int64 **)v50;
            v52 &= 0xFFFFFFFFFFFFFFF8LL;
            v25[3] = (__int64 **)(v52 | (unsigned __int8)v51 & 7);
            *(_QWORD *)(v52 + 8) = v25 + 3;
            *v50 = *v50 & 7 | (unsigned __int64)(v25 + 3);
          }
          sub_164B780((__int64)v25, v64);
          if ( v70 )
          {
            v63 = v70;
            sub_1623A60((__int64)&v63, (__int64)v70, 2);
            v53 = (__int64)v25[6];
            if ( v53 )
              sub_161E7C0((__int64)(v25 + 6), v53);
            v54 = v63;
            v25[6] = (__int64 **)v63;
            if ( v54 )
              sub_1623210((__int64)&v63, v54, (__int64)(v25 + 6));
          }
        }
        else
        {
          v25 = (__int64 ***)sub_15A46C0(47, *(__int64 ****)v14, v24, 0);
        }
      }
      if ( *((_BYTE *)v25 + 16) > 0x10u || *(_BYTE *)(a6 + 16) > 0x10u )
      {
        v69 = 257;
        v32 = sub_1648A60(56, 2u);
        v27 = (__int64)v32;
        if ( v32 )
        {
          v33 = (__int64)v32;
          v34 = *v25;
          if ( *((_BYTE *)*v25 + 8) == 16 )
          {
            v59 = v34[4];
            v35 = (__int64 *)sub_1643320(*v34);
            v36 = (__int64)sub_16463B0(v35, (unsigned int)v59);
          }
          else
          {
            v36 = sub_1643320(*v34);
          }
          sub_15FEC10(v27, v36, 51, v60, (__int64)v25, a6, (__int64)v68, 0);
        }
        else
        {
          v33 = 0;
        }
        if ( v71 )
        {
          v37 = (__int64 *)v72;
          sub_157E9D0(v71 + 40, v27);
          v38 = *(_QWORD *)(v27 + 24);
          v39 = *v37;
          *(_QWORD *)(v27 + 32) = v37;
          v39 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v27 + 24) = v39 | v38 & 7;
          *(_QWORD *)(v39 + 8) = v27 + 24;
          *v37 = *v37 & 7 | (v27 + 24);
        }
        sub_164B780(v33, v66);
        if ( v70 )
        {
          v63 = v70;
          sub_1623A60((__int64)&v63, (__int64)v70, 2);
          v40 = *(_QWORD *)(v27 + 48);
          if ( v40 )
            sub_161E7C0(v27 + 48, v40);
          v41 = v63;
          *(_QWORD *)(v27 + 48) = v63;
          if ( v41 )
            sub_1623210((__int64)&v63, v41, v27 + 48);
        }
      }
      else
      {
        v27 = sub_15A37B0(v60, v25, (_QWORD *)a6, 0);
      }
      v67 = 257;
      v30 = (__int64 ***)(*(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      v31 = *v30;
      if ( *v30 != *(__int64 ***)v27 )
      {
        if ( *(_BYTE *)(v27 + 16) > 0x10u )
        {
          v69 = 257;
          v42 = sub_15FDBD0(37, v27, (__int64)v31, (__int64)v68, 0);
          v27 = v42;
          if ( v71 )
          {
            v43 = (__int64 *)v72;
            sub_157E9D0(v71 + 40, v42);
            v44 = *(_QWORD *)(v27 + 24);
            v45 = *v43;
            *(_QWORD *)(v27 + 32) = v43;
            v45 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v27 + 24) = v45 | v44 & 7;
            *(_QWORD *)(v45 + 8) = v27 + 24;
            *v43 = *v43 & 7 | (v27 + 24);
          }
          sub_164B780(v27, v66);
          if ( v70 )
          {
            v64[0] = (__int64)v70;
            sub_1623A60((__int64)v64, (__int64)v70, 2);
            v46 = *(_QWORD *)(v27 + 48);
            if ( v46 )
              sub_161E7C0(v27 + 48, v46);
            v47 = (unsigned __int8 *)v64[0];
            *(_QWORD *)(v27 + 48) = v64[0];
            if ( v47 )
              sub_1623210((__int64)v64, v47, v27 + 48);
          }
        }
        else
        {
          v27 = sub_15A46C0(37, (__int64 ***)v27, v31, 0);
        }
      }
      if ( *(_BYTE *)(a1 + 80) )
        sub_18B6C20(
          v14,
          "unique-ret-val",
          14,
          a3,
          a4,
          v26,
          *(__int64 (__fastcall **)(__int64, __int64))(a1 + 88),
          *(_QWORD *)(a1 + 96));
      sub_164D160(*(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL, v27, a7, a8, a9, a10, v28, v29, a13, a14);
      v16 = *(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v16 + 16) == 29 )
      {
        v17 = *(_QWORD *)(v16 - 48);
        v18 = sub_1648A60(56, 1u);
        if ( v18 )
          sub_15F8320((__int64)v18, v17, v16);
        sub_157F2D0(*(_QWORD *)(v16 - 24), *(_QWORD *)(v16 + 40), 0);
        v16 = *(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      }
      sub_15F20C0((_QWORD *)v16);
      v19 = *(_DWORD **)(v14 + 16);
      if ( v19 )
        --*v19;
      if ( v70 )
        sub_161E7C0((__int64)&v70, (__int64)v70);
      v14 += 24;
    }
    while ( v61 != v14 );
  }
  *(_BYTE *)(a2 + 24) = 1;
  result = *(_QWORD *)(a2 + 32);
  if ( result != *(_QWORD *)(a2 + 40) )
    *(_QWORD *)(a2 + 40) = result;
  return result;
}
