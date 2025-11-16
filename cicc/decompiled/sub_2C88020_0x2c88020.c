// Function: sub_2C88020
// Address: 0x2c88020
//
__int64 __fastcall sub_2C88020(__int64 a1, __int64 a2)
{
  void *v2; // rax
  unsigned __int64 v3; // r8
  unsigned int v4; // r12d
  _BOOL4 v5; // eax
  _QWORD v7[4]; // [rsp+0h] [rbp-2D0h] BYREF
  int v8; // [rsp+20h] [rbp-2B0h] BYREF
  unsigned __int64 v9; // [rsp+28h] [rbp-2A8h]
  int *v10; // [rsp+30h] [rbp-2A0h]
  int *v11; // [rsp+38h] [rbp-298h]
  __int64 v12; // [rsp+40h] [rbp-290h]
  int v13; // [rsp+50h] [rbp-280h] BYREF
  unsigned __int64 v14; // [rsp+58h] [rbp-278h]
  int *v15; // [rsp+60h] [rbp-270h]
  int *v16; // [rsp+68h] [rbp-268h]
  __int64 v17; // [rsp+70h] [rbp-260h]
  int v18; // [rsp+80h] [rbp-250h] BYREF
  unsigned __int64 v19; // [rsp+88h] [rbp-248h]
  int *v20; // [rsp+90h] [rbp-240h]
  int *v21; // [rsp+98h] [rbp-238h]
  __int64 v22; // [rsp+A0h] [rbp-230h]
  int v23; // [rsp+B0h] [rbp-220h] BYREF
  unsigned __int64 v24; // [rsp+B8h] [rbp-218h]
  int *v25; // [rsp+C0h] [rbp-210h]
  int *v26; // [rsp+C8h] [rbp-208h]
  __int64 v27; // [rsp+D0h] [rbp-200h]
  int v28; // [rsp+E0h] [rbp-1F0h] BYREF
  unsigned __int64 v29; // [rsp+E8h] [rbp-1E8h]
  int *v30; // [rsp+F0h] [rbp-1E0h]
  int *v31; // [rsp+F8h] [rbp-1D8h]
  __int64 v32; // [rsp+100h] [rbp-1D0h]
  int v33; // [rsp+110h] [rbp-1C0h] BYREF
  unsigned __int64 v34; // [rsp+118h] [rbp-1B8h]
  int *v35; // [rsp+120h] [rbp-1B0h]
  int *v36; // [rsp+128h] [rbp-1A8h]
  __int64 v37; // [rsp+130h] [rbp-1A0h]
  int v38; // [rsp+140h] [rbp-190h] BYREF
  unsigned __int64 v39; // [rsp+148h] [rbp-188h]
  int *v40; // [rsp+150h] [rbp-180h]
  int *v41; // [rsp+158h] [rbp-178h]
  __int64 v42; // [rsp+160h] [rbp-170h]
  int v43; // [rsp+170h] [rbp-160h] BYREF
  unsigned __int64 v44; // [rsp+178h] [rbp-158h]
  int *v45; // [rsp+180h] [rbp-150h]
  int *v46; // [rsp+188h] [rbp-148h]
  __int64 v47; // [rsp+190h] [rbp-140h]
  int v48; // [rsp+1A0h] [rbp-130h] BYREF
  unsigned __int64 v49; // [rsp+1A8h] [rbp-128h]
  int *v50; // [rsp+1B0h] [rbp-120h]
  int *v51; // [rsp+1B8h] [rbp-118h]
  __int64 v52; // [rsp+1C0h] [rbp-110h]
  int v53; // [rsp+1D0h] [rbp-100h] BYREF
  unsigned __int64 v54; // [rsp+1D8h] [rbp-F8h]
  int *v55; // [rsp+1E0h] [rbp-F0h]
  int *v56; // [rsp+1E8h] [rbp-E8h]
  __int64 v57; // [rsp+1F0h] [rbp-E0h]
  int v58; // [rsp+200h] [rbp-D0h] BYREF
  unsigned __int64 v59; // [rsp+208h] [rbp-C8h]
  int *v60; // [rsp+210h] [rbp-C0h]
  int *v61; // [rsp+218h] [rbp-B8h]
  __int64 v62; // [rsp+220h] [rbp-B0h]
  int v63; // [rsp+230h] [rbp-A0h] BYREF
  unsigned __int64 v64; // [rsp+238h] [rbp-98h]
  int *v65; // [rsp+240h] [rbp-90h]
  int *v66; // [rsp+248h] [rbp-88h]
  __int64 v67; // [rsp+250h] [rbp-80h]
  int v68; // [rsp+260h] [rbp-70h] BYREF
  unsigned __int64 v69; // [rsp+268h] [rbp-68h]
  int *v70; // [rsp+270h] [rbp-60h]
  int *v71; // [rsp+278h] [rbp-58h]
  __int64 v72; // [rsp+280h] [rbp-50h]
  int v73; // [rsp+290h] [rbp-40h] BYREF
  unsigned __int64 v74; // [rsp+298h] [rbp-38h]
  int *v75; // [rsp+2A0h] [rbp-30h]
  int *v76; // [rsp+2A8h] [rbp-28h]
  __int64 v77; // [rsp+2B0h] [rbp-20h]

  v2 = sub_CB72A0();
  v3 = 0;
  v7[1] = 0;
  v7[0] = v2;
  v10 = &v8;
  v11 = &v8;
  v15 = &v13;
  v16 = &v13;
  v20 = &v18;
  v21 = &v18;
  v25 = &v23;
  v26 = &v23;
  v30 = &v28;
  v31 = &v28;
  v35 = &v33;
  v36 = &v33;
  v7[2] = 0;
  v8 = 0;
  v9 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v37 = 0;
  v40 = &v38;
  v41 = &v38;
  v45 = &v43;
  v46 = &v43;
  v50 = &v48;
  v51 = &v48;
  v55 = &v53;
  v56 = &v53;
  v60 = &v58;
  v61 = &v58;
  v65 = &v63;
  v66 = &v63;
  v38 = 0;
  v39 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = &v68;
  v4 = (unsigned __int8)qword_5011228;
  v71 = &v68;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = &v73;
  v76 = &v73;
  v77 = 0;
  if ( (_BYTE)qword_5011228 )
  {
    v5 = sub_2C84BA0(v7, a2);
    v3 = v74;
    v4 = v5;
  }
  sub_2C83D50(v3);
  sub_2C83D50(v69);
  sub_2C83D50(v64);
  sub_2C83D50(v59);
  sub_2C84080(v54);
  sub_2C84080(v49);
  sub_2C84080(v44);
  sub_2C84080(v39);
  sub_2C84080(v34);
  sub_2C84080(v29);
  sub_2C84080(v24);
  sub_2C84080(v19);
  sub_2C84080(v14);
  sub_2C84080(v9);
  return v4;
}
