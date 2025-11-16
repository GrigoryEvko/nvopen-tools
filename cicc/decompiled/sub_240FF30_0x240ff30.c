// Function: sub_240FF30
// Address: 0x240ff30
//
__int64 __fastcall sub_240FF30(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4, char a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  char **v15; // rsi
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // r10
  __int64 *v22; // rax
  unsigned __int64 v23; // rdx
  __int64 *v24; // r15
  __int64 v25; // r10
  unsigned int v26; // ebx
  int v27; // ecx
  __int64 v28; // r11
  _QWORD *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r13
  unsigned __int16 v32; // r14
  _QWORD *v33; // rdi
  __int64 v35; // r13
  _QWORD *v36; // rdi
  __int64 v37; // rax
  char *v38; // rax
  signed __int64 v39; // rdx
  __int64 v40; // r13
  _QWORD *v41; // rdi
  _QWORD *v42; // rdi
  int v43; // [rsp+10h] [rbp-180h]
  __int64 v44; // [rsp+10h] [rbp-180h]
  __int64 v45; // [rsp+18h] [rbp-178h]
  __int64 v46; // [rsp+20h] [rbp-170h]
  __int64 v47; // [rsp+28h] [rbp-168h]
  __int64 v48; // [rsp+28h] [rbp-168h]
  __int64 v49; // [rsp+28h] [rbp-168h]
  __int64 v50; // [rsp+30h] [rbp-160h]
  unsigned __int16 v51; // [rsp+30h] [rbp-160h]
  __int64 v52; // [rsp+38h] [rbp-158h]
  unsigned __int16 v53; // [rsp+38h] [rbp-158h]
  __int64 v55; // [rsp+40h] [rbp-150h]
  unsigned int v56; // [rsp+48h] [rbp-148h]
  __int64 v57; // [rsp+48h] [rbp-148h]
  __int64 v58; // [rsp+58h] [rbp-138h] BYREF
  __int64 v59; // [rsp+60h] [rbp-130h] BYREF
  unsigned __int16 v60; // [rsp+68h] [rbp-128h]
  char v61[32]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v62; // [rsp+90h] [rbp-100h]
  __int64 v63; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned __int16 v64; // [rsp+A8h] [rbp-E8h]
  __int16 v65; // [rsp+C0h] [rbp-D0h]
  char *v66; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+D8h] [rbp-B8h]
  _BYTE v68[16]; // [rsp+E0h] [rbp-B0h] BYREF
  _QWORD *v69; // [rsp+F0h] [rbp-A0h]
  __int64 v70; // [rsp+100h] [rbp-90h]
  __int64 v71; // [rsp+108h] [rbp-88h]
  __int16 v72; // [rsp+110h] [rbp-80h]
  __int64 v73; // [rsp+118h] [rbp-78h]
  void **v74; // [rsp+120h] [rbp-70h]
  void **v75; // [rsp+128h] [rbp-68h]
  __int64 v76; // [rsp+130h] [rbp-60h]
  int v77; // [rsp+138h] [rbp-58h]
  __int16 v78; // [rsp+13Ch] [rbp-54h]
  char v79; // [rsp+13Eh] [rbp-52h]
  __int64 v80; // [rsp+140h] [rbp-50h]
  __int64 v81; // [rsp+148h] [rbp-48h]
  void *v82; // [rsp+150h] [rbp-40h] BYREF
  void *v83; // [rsp+158h] [rbp-38h] BYREF

  v9 = a2[3];
  LOWORD(v69) = 261;
  v10 = a2[5];
  v52 = v9;
  v11 = a2[1];
  v66 = (char *)a3;
  v67 = a4;
  v50 = v10;
  v56 = *(_DWORD *)(v11 + 8) >> 8;
  v12 = sub_BD2DA0(136);
  v13 = v12;
  if ( v12 )
    sub_B2C3B0(v12, a6, a5, v56, (__int64)&v66, v50);
  sub_B2EC90(v13, (__int64)a2);
  v63 = *(_QWORD *)(v13 + 120);
  v14 = sub_A74610(&v63);
  sub_A751C0((__int64)&v66, **(_QWORD **)(a6 + 16), v14, 3);
  v15 = &v66;
  sub_B2D550(v13, (__int64)&v66);
  sub_240DFF0(v69);
  v66 = "entry";
  LOWORD(v69) = 259;
  v16 = a1[1];
  v17 = sub_22077B0(0x50u);
  v57 = v17;
  if ( v17 )
  {
    v15 = (char **)v16;
    sub_AA4D50(v17, v16, (__int64)&v66, v13, 0);
  }
  if ( *(_DWORD *)(a2[3] + 8LL) >> 8 )
  {
    sub_B2D4A0(v13, "split-stack", 0xBu);
    sub_B43C20((__int64)&v59, v57);
    v65 = 257;
    v37 = sub_AA48A0(v57);
    v75 = &v83;
    v73 = v37;
    v67 = 0x200000000LL;
    v66 = v68;
    v82 = &unk_49DA100;
    v78 = 512;
    v62 = 257;
    v83 = &unk_49DA0B0;
    v72 = 0;
    v70 = v57;
    v74 = &v82;
    v76 = 0;
    v77 = 0;
    v79 = 7;
    v80 = 0;
    v81 = 0;
    v71 = v57 + 48;
    v38 = (char *)sub_BD5D20((__int64)a2);
    v58 = sub_B33830((__int64)&v66, v38, v39, (__int64)v61, 0, 0, 1);
    v53 = v60;
    v40 = a1[47];
    v44 = v59;
    v49 = a1[48];
    v41 = sub_BD2C40(88, 2u);
    if ( v41 )
    {
      sub_B44260((__int64)v41, **(_QWORD **)(v40 + 16), 56, 2u, v44, v53);
      v41[9] = 0;
      sub_B4A290((__int64)v41, v40, v49, &v58, 1, (__int64)&v63, 0, 0);
    }
    nullsub_61();
    v82 = &unk_49DA100;
    nullsub_63();
    if ( v66 != v68 )
      _libc_free((unsigned __int64)v66);
    sub_B43C20((__int64)&v66, v57);
    v42 = sub_BD2C40(72, unk_3F148B8);
    if ( v42 )
      sub_B4C8A0((__int64)v42, a1[1], (__int64)v66, v67);
  }
  else
  {
    if ( (*(_BYTE *)(v13 + 2) & 1) != 0 )
      sub_B2C6D0(v13, (__int64)v15, v18, v19);
    v20 = *(_QWORD *)(v13 + 96);
    v47 = 40LL * (unsigned int)(*(_DWORD *)(v52 + 12) - 1);
    v21 = v47;
    if ( v47 )
    {
      v22 = (__int64 *)sub_22077B0(0x6666666666666668LL * (v47 >> 3));
      v23 = 0xCCCCCCCCCCCCCCCDLL * (v47 >> 3);
      v24 = v22;
      do
      {
        *v22++ = v20;
        v20 += 40;
        --v23;
      }
      while ( v23 );
      v25 = 8;
      if ( v47 > 0 )
        v25 = 0x6666666666666668LL * (v47 >> 3);
      v21 = v25 >> 3;
      v26 = v21 + 1;
      v27 = (v21 + 1) & 0x7FFFFFF;
    }
    else
    {
      v27 = 1;
      v26 = 1;
      v24 = 0;
    }
    v43 = v27;
    v45 = v21;
    sub_B43C20((__int64)&v63, v57);
    v28 = a2[3];
    LOWORD(v69) = 257;
    v48 = v28;
    v46 = v63;
    v51 = v64;
    v29 = sub_BD2C40(88, v26);
    v30 = (__int64)v29;
    if ( v29 )
    {
      sub_B44260((__int64)v29, **(_QWORD **)(v48 + 16), 56, v43 & 0x1FFFFFFF, v46, v51);
      *(_QWORD *)(v30 + 72) = 0;
      sub_B4A290(v30, v48, (__int64)a2, v24, v45, (__int64)&v66, 0, 0);
    }
    if ( *(_BYTE *)(**(_QWORD **)(v52 + 16) + 8LL) == 7 )
    {
      v35 = a1[1];
      sub_B43C20((__int64)&v66, v57);
      v36 = sub_BD2C40(72, 0);
      if ( v36 )
        sub_B4BB80((__int64)v36, v35, 0, 0, (__int64)v66, v67);
    }
    else
    {
      sub_B43C20((__int64)&v66, v57);
      v31 = a1[1];
      v32 = v67;
      v55 = (__int64)v66;
      v33 = sub_BD2C40(72, v30 != 0);
      if ( v33 )
        sub_B4BB80((__int64)v33, v31, v30, v30 != 0, v55, v32);
    }
    if ( v24 )
      j_j___libc_free_0((unsigned __int64)v24);
  }
  return v13;
}
