// Function: sub_166CBC0
// Address: 0x166cbc0
//
__int64 __fastcall sub_166CBC0(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  bool v6; // zf
  __int64 *v7; // rbx
  int v8; // r13d
  __int64 v9; // rsi
  int v10; // eax
  unsigned int v11; // r13d
  _QWORD v14[2]; // [rsp+10h] [rbp-6C0h] BYREF
  _BYTE v15[40]; // [rsp+20h] [rbp-6B0h] BYREF
  __int64 v16; // [rsp+48h] [rbp-688h]
  __int64 v17; // [rsp+50h] [rbp-680h]
  __int16 v18; // [rsp+58h] [rbp-678h]
  bool v19; // [rsp+5Ah] [rbp-676h]
  char *v20; // [rsp+60h] [rbp-670h]
  __int64 v21; // [rsp+68h] [rbp-668h]
  char v22; // [rsp+70h] [rbp-660h] BYREF
  __int64 v23; // [rsp+78h] [rbp-658h]
  __int64 v24; // [rsp+80h] [rbp-650h]
  __int64 v25; // [rsp+88h] [rbp-648h]
  int v26; // [rsp+90h] [rbp-640h]
  __int64 v27; // [rsp+A0h] [rbp-630h]
  char v28; // [rsp+A8h] [rbp-628h]
  int v29; // [rsp+ACh] [rbp-624h]
  __int64 v30; // [rsp+B0h] [rbp-620h]
  _BYTE *v31; // [rsp+B8h] [rbp-618h]
  _BYTE *v32; // [rsp+C0h] [rbp-610h]
  __int64 v33; // [rsp+C8h] [rbp-608h]
  int v34; // [rsp+D0h] [rbp-600h]
  _BYTE v35[128]; // [rsp+D8h] [rbp-5F8h] BYREF
  __int64 v36; // [rsp+158h] [rbp-578h]
  _BYTE *v37; // [rsp+160h] [rbp-570h]
  _BYTE *v38; // [rsp+168h] [rbp-568h]
  __int64 v39; // [rsp+170h] [rbp-560h]
  int v40; // [rsp+178h] [rbp-558h]
  _BYTE v41[256]; // [rsp+180h] [rbp-550h] BYREF
  __int64 v42; // [rsp+280h] [rbp-450h]
  __int64 v43; // [rsp+288h] [rbp-448h]
  __int64 v44; // [rsp+290h] [rbp-440h]
  int v45; // [rsp+298h] [rbp-438h]
  __int64 v46; // [rsp+2A0h] [rbp-430h]
  _BYTE *v47; // [rsp+2A8h] [rbp-428h]
  _BYTE *v48; // [rsp+2B0h] [rbp-420h]
  __int64 v49; // [rsp+2B8h] [rbp-418h]
  int v50; // [rsp+2C0h] [rbp-410h]
  _BYTE v51[16]; // [rsp+2C8h] [rbp-408h] BYREF
  __int64 v52; // [rsp+2D8h] [rbp-3F8h]
  __int16 v53; // [rsp+2E0h] [rbp-3F0h]
  __int64 v54; // [rsp+2E8h] [rbp-3E8h]
  __int64 v55; // [rsp+2F0h] [rbp-3E0h]
  __int64 v56; // [rsp+2F8h] [rbp-3D8h]
  int v57; // [rsp+300h] [rbp-3D0h]
  __int64 v58; // [rsp+308h] [rbp-3C8h]
  __int64 v59; // [rsp+310h] [rbp-3C0h]
  __int64 v60; // [rsp+318h] [rbp-3B8h]
  int v61; // [rsp+320h] [rbp-3B0h]
  __int64 v62; // [rsp+328h] [rbp-3A8h]
  __int64 v63; // [rsp+330h] [rbp-3A0h]
  __int64 v64; // [rsp+338h] [rbp-398h]
  __int64 v65; // [rsp+340h] [rbp-390h]
  _BYTE *v66; // [rsp+348h] [rbp-388h]
  _BYTE *v67; // [rsp+350h] [rbp-380h]
  __int64 v68; // [rsp+358h] [rbp-378h]
  int v69; // [rsp+360h] [rbp-370h]
  _BYTE v70[256]; // [rsp+368h] [rbp-368h] BYREF
  char *v71; // [rsp+468h] [rbp-268h]
  __int64 v72; // [rsp+470h] [rbp-260h]
  char v73; // [rsp+478h] [rbp-258h] BYREF
  __int64 v74; // [rsp+498h] [rbp-238h]
  _BYTE *v75; // [rsp+4A0h] [rbp-230h]
  _BYTE *v76; // [rsp+4A8h] [rbp-228h]
  __int64 v77; // [rsp+4B0h] [rbp-220h]
  int v78; // [rsp+4B8h] [rbp-218h]
  _BYTE v79[256]; // [rsp+4C0h] [rbp-210h] BYREF
  char *v80; // [rsp+5C0h] [rbp-110h]
  __int64 v81; // [rsp+5C8h] [rbp-108h]
  char v82; // [rsp+5D0h] [rbp-100h] BYREF
  _QWORD *v83; // [rsp+650h] [rbp-80h]
  __int64 v84; // [rsp+658h] [rbp-78h]
  __int64 v85; // [rsp+660h] [rbp-70h]
  __int64 v86; // [rsp+668h] [rbp-68h]
  int v87; // [rsp+670h] [rbp-60h]
  __int64 v88; // [rsp+678h] [rbp-58h]
  __int64 v89; // [rsp+680h] [rbp-50h]
  __int64 v90; // [rsp+688h] [rbp-48h]
  int i; // [rsp+690h] [rbp-40h]

  v14[0] = a2;
  v14[1] = a1;
  sub_154BA10((__int64)v15, (__int64)a1, 1);
  v4 = sub_1632FA0((__int64)a1);
  v28 = 0;
  v16 = v4;
  v5 = *a1;
  v23 = 0;
  v17 = v5;
  v6 = a3 == 0;
  v18 = 0;
  v20 = &v22;
  v21 = 0x100000000LL;
  v31 = v35;
  v32 = v35;
  v37 = v41;
  v38 = v41;
  v47 = v51;
  v48 = v51;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v29 = 0;
  v30 = 0;
  v33 = 16;
  v34 = 0;
  v36 = 0;
  v39 = 32;
  v40 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v49 = 2;
  v50 = 0;
  v52 = 0;
  v66 = v70;
  v67 = v70;
  v71 = &v73;
  v72 = 0x400000000LL;
  v75 = v79;
  v76 = v79;
  v80 = &v82;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v68 = 32;
  v69 = 0;
  v74 = 0;
  v77 = 32;
  v78 = 0;
  v81 = 0x1000000000LL;
  v83 = v14;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v7 = (__int64 *)a1[4];
  v19 = v6;
  v8 = 0;
  v90 = 0;
  for ( i = 0; a1 + 3 != v7; v8 |= v10 ^ 1 )
  {
    v9 = (__int64)(v7 - 7);
    if ( !v7 )
      v9 = 0;
    v10 = sub_166A310(v14, v9);
    v7 = (__int64 *)v7[1];
  }
  v11 = sub_165D700((__int64)v14) ^ 1 | v8;
  if ( a3 )
    *a3 = HIBYTE(v18);
  sub_164DC80((__int64)v14);
  return v11;
}
