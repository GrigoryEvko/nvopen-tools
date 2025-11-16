// Function: sub_2397950
// Address: 0x2397950
//
_QWORD *__fastcall sub_2397950(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // zf
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  _BYTE v33[96]; // [rsp+0h] [rbp-2F0h] BYREF
  _BYTE v34[96]; // [rsp+60h] [rbp-290h] BYREF
  __int64 v35; // [rsp+C0h] [rbp-230h]
  __int64 v36; // [rsp+C8h] [rbp-228h]
  __int64 v37; // [rsp+D0h] [rbp-220h]
  __int64 v38; // [rsp+D8h] [rbp-218h]
  int v39; // [rsp+E0h] [rbp-210h]
  __int64 v40; // [rsp+E8h] [rbp-208h]
  __int64 v41; // [rsp+F0h] [rbp-200h]
  __int64 v42; // [rsp+F8h] [rbp-1F8h]
  int v43; // [rsp+100h] [rbp-1F0h]
  int v44; // [rsp+108h] [rbp-1E8h]
  __int64 v45; // [rsp+110h] [rbp-1E0h]
  __int64 v46; // [rsp+118h] [rbp-1D8h]
  __int64 v47; // [rsp+120h] [rbp-1D0h]
  int v48; // [rsp+128h] [rbp-1C8h]
  int v49; // [rsp+130h] [rbp-1C0h]
  char v50; // [rsp+134h] [rbp-1BCh]
  __int64 v51; // [rsp+138h] [rbp-1B8h]
  __int64 v52; // [rsp+140h] [rbp-1B0h]
  __int64 v53; // [rsp+148h] [rbp-1A8h]
  char v54; // [rsp+150h] [rbp-1A0h]
  _BYTE v55[96]; // [rsp+160h] [rbp-190h] BYREF
  _BYTE v56[96]; // [rsp+1C0h] [rbp-130h] BYREF
  __int64 v57; // [rsp+220h] [rbp-D0h]
  __int64 v58; // [rsp+228h] [rbp-C8h]
  __int64 v59; // [rsp+230h] [rbp-C0h]
  __int64 v60; // [rsp+238h] [rbp-B8h]
  int v61; // [rsp+240h] [rbp-B0h]
  __int64 v62; // [rsp+248h] [rbp-A8h]
  __int64 v63; // [rsp+250h] [rbp-A0h]
  __int64 v64; // [rsp+258h] [rbp-98h]
  int v65; // [rsp+260h] [rbp-90h]
  int v66; // [rsp+268h] [rbp-88h]
  __int64 v67; // [rsp+270h] [rbp-80h]
  __int64 v68; // [rsp+278h] [rbp-78h]
  __int64 v69; // [rsp+280h] [rbp-70h]
  int v70; // [rsp+288h] [rbp-68h]
  int v71; // [rsp+290h] [rbp-60h]
  char v72; // [rsp+294h] [rbp-5Ch]
  __int64 v73; // [rsp+298h] [rbp-58h]
  __int64 v74; // [rsp+2A0h] [rbp-50h]
  __int64 v75; // [rsp+2A8h] [rbp-48h]
  char v76; // [rsp+2B0h] [rbp-40h]

  sub_22BD750((__int64)v33, a2 + 8, a3);
  sub_234E5E0((__int64)v55, (__int64)v33, v3, v4, v5, v6);
  sub_234E5E0((__int64)v56, (__int64)v34, v7, v8, v9, v10);
  ++v36;
  ++v40;
  v57 = v35;
  v58 = 1;
  v59 = v37;
  v37 = 0;
  v60 = v38;
  v62 = 1;
  v38 = 0;
  v61 = v39;
  v39 = 0;
  v63 = v41;
  v41 = 0;
  v64 = v42;
  v42 = 0;
  v11 = v43;
  v43 = 0;
  v65 = v11;
  v66 = v44;
  v76 = 0;
  v67 = v45;
  v68 = v46;
  v69 = v47;
  v70 = v48;
  v71 = v49;
  v72 = v50;
  if ( v54 )
  {
    v12 = v51;
    v76 = 1;
    v51 = 0;
    v73 = v12;
    v13 = v52;
    v52 = 0;
    v74 = v13;
    v14 = v53;
    v53 = 0;
    v75 = v14;
  }
  v15 = (_QWORD *)sub_22077B0(0x160u);
  v20 = v15;
  if ( v15 )
  {
    *v15 = &unk_4A0B3A8;
    sub_234E5E0((__int64)(v15 + 1), (__int64)v55, v16, v17, v18, v19);
    sub_234E5E0((__int64)(v20 + 13), (__int64)v56, v21, v22, v23, v24);
    ++v58;
    ++v62;
    v20[25] = v57;
    v25 = v59;
    v20[26] = 1;
    v20[27] = v25;
    v59 = 0;
    v20[28] = v60;
    v60 = 0;
    *((_DWORD *)v20 + 58) = v61;
    v61 = 0;
    v20[31] = v63;
    v26 = v64;
    v20[30] = 1;
    v20[32] = v26;
    v63 = 0;
    *((_DWORD *)v20 + 66) = v65;
    v64 = 0;
    *((_DWORD *)v20 + 68) = v66;
    v65 = 0;
    v20[35] = v67;
    v20[36] = v68;
    v27 = v69;
    v28 = v76 == 0;
    *((_BYTE *)v20 + 344) = 0;
    v20[37] = v27;
    *((_DWORD *)v20 + 76) = v70;
    *((_DWORD *)v20 + 78) = v71;
    *((_BYTE *)v20 + 316) = v72;
    if ( !v28 )
    {
      v29 = v73;
      *((_BYTE *)v20 + 344) = 1;
      v73 = 0;
      v20[40] = v29;
      v30 = v74;
      v74 = 0;
      v20[41] = v30;
      v31 = v75;
      v75 = 0;
      v20[42] = v31;
    }
  }
  sub_2397770((__int64)v55);
  *a1 = v20;
  sub_2397770((__int64)v33);
  return a1;
}
