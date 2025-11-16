// Function: sub_1C8A5C0
// Address: 0x1c8a5c0
//
unsigned __int64 __fastcall sub_1C8A5C0(
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
  __int64 ****v13; // rax
  __int64 ***v14; // r12
  unsigned __int64 *v15; // r13
  char v16; // bl
  unsigned __int64 result; // rax
  __int64 v18; // rax
  unsigned __int8 *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 **v22; // rcx
  __int64 *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r11
  __int64 v27; // r12
  double v28; // xmm4_8
  double v29; // xmm5_8
  unsigned __int8 *v30; // rsi
  __int64 **v31; // rax
  __int64 v32; // r11
  __int64 v33; // rax
  __int64 **v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 **v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r11
  unsigned __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned __int8 *v45; // rsi
  __int64 v46; // rax
  __int64 v47; // r11
  unsigned __int64 *v48; // r12
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned __int8 *v53; // rsi
  __int64 *v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rsi
  unsigned __int8 *v58; // rsi
  __int64 v59; // [rsp+0h] [rbp-160h]
  __int64 v60; // [rsp+8h] [rbp-158h]
  __int64 **v61; // [rsp+8h] [rbp-158h]
  __int64 v62; // [rsp+10h] [rbp-150h]
  _QWORD *v63; // [rsp+18h] [rbp-148h]
  char v64; // [rsp+20h] [rbp-140h]
  __int64 v66; // [rsp+28h] [rbp-138h]
  __int64 v68; // [rsp+30h] [rbp-130h]
  __int64 v69; // [rsp+30h] [rbp-130h]
  __int64 v70; // [rsp+30h] [rbp-130h]
  unsigned __int64 *v71; // [rsp+30h] [rbp-130h]
  __int64 v72; // [rsp+30h] [rbp-130h]
  __int64 v73; // [rsp+30h] [rbp-130h]
  __int64 v74; // [rsp+30h] [rbp-130h]
  __int64 v75; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v77; // [rsp+48h] [rbp-118h] BYREF
  _QWORD v78[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v79; // [rsp+60h] [rbp-100h] BYREF
  __int16 v80; // [rsp+70h] [rbp-F0h]
  __int64 v81[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v82; // [rsp+90h] [rbp-D0h]
  __int64 v83[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v84; // [rsp+B0h] [rbp-B0h]
  unsigned __int8 *v85[2]; // [rsp+C0h] [rbp-A0h] BYREF
  _QWORD v86[2]; // [rsp+D0h] [rbp-90h] BYREF
  unsigned __int8 *v87; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v88; // [rsp+E8h] [rbp-78h]
  unsigned __int64 *v89; // [rsp+F0h] [rbp-70h]
  __int64 v90; // [rsp+F8h] [rbp-68h]
  __int64 v91; // [rsp+100h] [rbp-60h]
  int v92; // [rsp+108h] [rbp-58h]
  __int64 v93; // [rsp+110h] [rbp-50h]
  __int64 v94; // [rsp+118h] [rbp-48h]

  if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
    v13 = (__int64 ****)*(a2 - 1);
  else
    v13 = (__int64 ****)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
  v14 = *v13;
  v15 = (unsigned __int64 *)v13[3];
  v16 = *(_BYTE *)(*a2 + 8LL);
  result = (unsigned __int64)**v13;
  if ( *(_BYTE *)(result + 8) == 5 && (result = *v15, *(_BYTE *)(*v15 + 8) == 5) )
  {
    v64 = 1;
  }
  else
  {
    if ( v16 != 5 )
      return result;
    v64 = 0;
  }
  v63 = (_QWORD *)sub_16498A0((__int64)a2);
  v62 = *(_QWORD *)(*(_QWORD *)(a2[5] + 56LL) + 40LL);
  v18 = sub_16498A0((__int64)a2);
  v19 = (unsigned __int8 *)a2[6];
  v87 = 0;
  v90 = v18;
  v20 = a2[5];
  v91 = 0;
  v88 = v20;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v89 = a2 + 3;
  v85[0] = v19;
  if ( v19 )
  {
    sub_1623A60((__int64)v85, (__int64)v19, 2);
    if ( v87 )
      sub_161E7C0((__int64)&v87, (__int64)v87);
    v87 = v85[0];
    if ( v85[0] )
      sub_1623210((__int64)v85, v85[0], (__int64)&v87);
  }
  if ( v64 )
  {
    v60 = sub_1643370(v63);
    v36 = sub_1643370(v63);
    v21 = v60;
    v22 = (__int64 **)v36;
  }
  else
  {
    v21 = *v15;
    v22 = *v14;
  }
  if ( v16 == 5 )
  {
    v59 = v21;
    v61 = v22;
    v38 = sub_1643370(v63);
    v22 = v61;
    v21 = v59;
    v23 = (__int64 *)v38;
  }
  else
  {
    v23 = (__int64 *)*a2;
  }
  v86[0] = v22;
  v86[1] = v21;
  v85[0] = (unsigned __int8 *)v86;
  v85[1] = (unsigned __int8 *)0x200000002LL;
  v24 = sub_1644EA0(v23, v86, 2, 0);
  v25 = sub_1632080(v62, a3, a4, v24, 0);
  v26 = v25;
  if ( (_QWORD *)v85[0] != v86 )
  {
    v68 = v25;
    _libc_free((unsigned __int64)v85[0]);
    v26 = v68;
  }
  v84 = 257;
  if ( v64 )
  {
    v69 = v26;
    v80 = 257;
    v31 = (__int64 **)sub_1643370(v63);
    v32 = v69;
    if ( v31 != *v14 )
    {
      if ( *((_BYTE *)v14 + 16) > 0x10u )
      {
        LOWORD(v86[0]) = 257;
        v39 = sub_15FDBD0(47, (__int64)v14, (__int64)v31, (__int64)v85, 0);
        v40 = v69;
        v14 = (__int64 ***)v39;
        if ( v88 )
        {
          v66 = v69;
          v71 = v89;
          sub_157E9D0(v88 + 40, v39);
          v40 = v66;
          v41 = *v71;
          v42 = (unsigned __int64)v14[3] & 7;
          v14[4] = (__int64 **)v71;
          v41 &= 0xFFFFFFFFFFFFFFF8LL;
          v14[3] = (__int64 **)(v41 | v42);
          *(_QWORD *)(v41 + 8) = v14 + 3;
          *v71 = *v71 & 7 | (unsigned __int64)(v14 + 3);
        }
        v72 = v40;
        sub_164B780((__int64)v14, &v79);
        v32 = v72;
        if ( v87 )
        {
          v81[0] = (__int64)v87;
          sub_1623A60((__int64)v81, (__int64)v87, 2);
          v43 = (__int64)v14[6];
          v44 = (__int64)(v14 + 6);
          v32 = v72;
          if ( v43 )
          {
            sub_161E7C0((__int64)(v14 + 6), v43);
            v32 = v72;
            v44 = (__int64)(v14 + 6);
          }
          v45 = (unsigned __int8 *)v81[0];
          v14[6] = (__int64 **)v81[0];
          if ( v45 )
          {
            v73 = v32;
            sub_1623210((__int64)v81, v45, v44);
            v32 = v73;
          }
        }
      }
      else
      {
        v33 = sub_15A46C0(47, v14, v31, 0);
        v32 = v69;
        v14 = (__int64 ***)v33;
      }
    }
    v70 = v32;
    v78[0] = v14;
    v82 = 257;
    v34 = (__int64 **)sub_1643370(v63);
    v26 = v70;
    if ( v34 != (__int64 **)*v15 )
    {
      if ( *((_BYTE *)v15 + 16) > 0x10u )
      {
        LOWORD(v86[0]) = 257;
        v46 = sub_15FDBD0(47, (__int64)v15, (__int64)v34, (__int64)v85, 0);
        v47 = v70;
        v15 = (unsigned __int64 *)v46;
        if ( v88 )
        {
          v48 = v89;
          sub_157E9D0(v88 + 40, v46);
          v49 = v15[3];
          v47 = v70;
          v50 = *v48;
          v15[4] = (unsigned __int64)v48;
          v50 &= 0xFFFFFFFFFFFFFFF8LL;
          v15[3] = v50 | v49 & 7;
          *(_QWORD *)(v50 + 8) = v15 + 3;
          *v48 = *v48 & 7 | (unsigned __int64)(v15 + 3);
        }
        v74 = v47;
        sub_164B780((__int64)v15, v81);
        v26 = v74;
        if ( v87 )
        {
          v77 = v87;
          sub_1623A60((__int64)&v77, (__int64)v87, 2);
          v51 = v15[6];
          v26 = v74;
          v52 = (__int64)(v15 + 6);
          if ( v51 )
          {
            sub_161E7C0((__int64)(v15 + 6), v51);
            v26 = v74;
            v52 = (__int64)(v15 + 6);
          }
          v53 = v77;
          v15[6] = (unsigned __int64)v77;
          if ( v53 )
          {
            v75 = v26;
            sub_1623210((__int64)&v77, v53, v52);
            v26 = v75;
          }
        }
      }
      else
      {
        v35 = sub_15A46C0(47, (__int64 ***)v15, v34, 0);
        v26 = v70;
        v15 = (unsigned __int64 *)v35;
      }
    }
  }
  else
  {
    v78[0] = v14;
  }
  v78[1] = v15;
  v27 = sub_1285290((__int64 *)&v87, *(_QWORD *)(*(_QWORD *)v26 + 24LL), v26, (int)v78, 2, (__int64)v83, 0);
  if ( v16 == 5 )
  {
    v84 = 257;
    v37 = (__int64 **)sub_16432F0(v63);
    if ( v37 != *(__int64 ***)v27 )
    {
      if ( *(_BYTE *)(v27 + 16) > 0x10u )
      {
        LOWORD(v86[0]) = 257;
        v27 = sub_15FDBD0(47, v27, (__int64)v37, (__int64)v85, 0);
        if ( v88 )
        {
          v54 = (__int64 *)v89;
          sub_157E9D0(v88 + 40, v27);
          v55 = *(_QWORD *)(v27 + 24);
          v56 = *v54;
          *(_QWORD *)(v27 + 32) = v54;
          v56 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v27 + 24) = v56 | v55 & 7;
          *(_QWORD *)(v56 + 8) = v27 + 24;
          *v54 = *v54 & 7 | (v27 + 24);
        }
        sub_164B780(v27, v83);
        if ( v87 )
        {
          v81[0] = (__int64)v87;
          sub_1623A60((__int64)v81, (__int64)v87, 2);
          v57 = *(_QWORD *)(v27 + 48);
          if ( v57 )
            sub_161E7C0(v27 + 48, v57);
          v58 = (unsigned __int8 *)v81[0];
          *(_QWORD *)(v27 + 48) = v81[0];
          if ( v58 )
            sub_1623210((__int64)v81, v58, v27 + 48);
        }
      }
      else
      {
        v27 = sub_15A46C0(47, (__int64 ***)v27, v37, 0);
      }
    }
  }
  sub_164D160((__int64)a2, v27, a5, a6, a7, a8, v28, v29, a11, a12);
  sub_15F20C0(a2);
  result = (unsigned __int64)a1;
  v30 = v87;
  *a1 = 1;
  if ( v30 )
    return sub_161E7C0((__int64)&v87, (__int64)v30);
  return result;
}
