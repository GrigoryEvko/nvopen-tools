// Function: sub_C08EC0
// Address: 0xc08ec0
//
__int64 __fastcall sub_C08EC0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-930h] BYREF
  _BYTE v6[112]; // [rsp+10h] [rbp-920h] BYREF
  _QWORD *v7; // [rsp+80h] [rbp-8B0h]
  _QWORD *v8; // [rsp+88h] [rbp-8A8h]
  __int64 v9; // [rsp+90h] [rbp-8A0h]
  __int16 v10; // [rsp+98h] [rbp-898h]
  char v11; // [rsp+9Ah] [rbp-896h]
  char *v12; // [rsp+A0h] [rbp-890h]
  __int64 v13; // [rsp+A8h] [rbp-888h]
  char v14; // [rsp+B0h] [rbp-880h] BYREF
  char *v15; // [rsp+B8h] [rbp-878h]
  __int64 v16; // [rsp+C0h] [rbp-870h]
  char v17; // [rsp+C8h] [rbp-868h] BYREF
  __int64 v18; // [rsp+100h] [rbp-830h]
  __int64 v19; // [rsp+108h] [rbp-828h]
  char v20; // [rsp+110h] [rbp-820h]
  __int64 v21; // [rsp+114h] [rbp-81Ch]
  __int64 v22; // [rsp+120h] [rbp-810h]
  char *v23; // [rsp+128h] [rbp-808h]
  __int64 v24; // [rsp+130h] [rbp-800h]
  int v25; // [rsp+138h] [rbp-7F8h]
  char v26; // [rsp+13Ch] [rbp-7F4h]
  char v27; // [rsp+140h] [rbp-7F0h] BYREF
  __int64 v28; // [rsp+1C0h] [rbp-770h]
  char *v29; // [rsp+1C8h] [rbp-768h]
  __int64 v30; // [rsp+1D0h] [rbp-760h]
  int v31; // [rsp+1D8h] [rbp-758h]
  char v32; // [rsp+1DCh] [rbp-754h]
  char v33; // [rsp+1E0h] [rbp-750h] BYREF
  __int64 v34; // [rsp+2E0h] [rbp-650h]
  __int64 v35; // [rsp+2E8h] [rbp-648h]
  __int64 v36; // [rsp+2F0h] [rbp-640h]
  int v37; // [rsp+2F8h] [rbp-638h]
  __int64 v38; // [rsp+300h] [rbp-630h]
  char *v39; // [rsp+308h] [rbp-628h]
  __int64 v40; // [rsp+310h] [rbp-620h]
  int v41; // [rsp+318h] [rbp-618h]
  char v42; // [rsp+31Ch] [rbp-614h]
  char v43; // [rsp+320h] [rbp-610h] BYREF
  __int64 v44; // [rsp+330h] [rbp-600h]
  __int16 v45; // [rsp+338h] [rbp-5F8h]
  __int64 v46; // [rsp+340h] [rbp-5F0h]
  __int64 v47; // [rsp+348h] [rbp-5E8h]
  __int64 v48; // [rsp+350h] [rbp-5E0h]
  int v49; // [rsp+358h] [rbp-5D8h]
  __int64 v50; // [rsp+360h] [rbp-5D0h]
  __int64 v51; // [rsp+368h] [rbp-5C8h]
  __int64 v52; // [rsp+370h] [rbp-5C0h]
  int v53; // [rsp+378h] [rbp-5B8h]
  _QWORD *v54; // [rsp+380h] [rbp-5B0h]
  __int64 v55; // [rsp+388h] [rbp-5A8h]
  _QWORD v56[3]; // [rsp+390h] [rbp-5A0h] BYREF
  int v57; // [rsp+3A8h] [rbp-588h]
  __int64 v58; // [rsp+3B0h] [rbp-580h]
  char *v59; // [rsp+3B8h] [rbp-578h]
  __int64 v60; // [rsp+3C0h] [rbp-570h]
  int v61; // [rsp+3C8h] [rbp-568h]
  char v62; // [rsp+3CCh] [rbp-564h]
  char v63; // [rsp+3D0h] [rbp-560h] BYREF
  char *v64; // [rsp+4D0h] [rbp-460h]
  __int64 v65; // [rsp+4D8h] [rbp-458h]
  char v66; // [rsp+4E0h] [rbp-450h] BYREF
  __int64 v67; // [rsp+500h] [rbp-430h]
  char *v68; // [rsp+508h] [rbp-428h]
  __int64 v69; // [rsp+510h] [rbp-420h]
  int v70; // [rsp+518h] [rbp-418h]
  char v71; // [rsp+51Ch] [rbp-414h]
  char v72; // [rsp+520h] [rbp-410h] BYREF
  __int64 v73; // [rsp+620h] [rbp-310h]
  char *v74; // [rsp+628h] [rbp-308h]
  __int64 v75; // [rsp+630h] [rbp-300h]
  int v76; // [rsp+638h] [rbp-2F8h]
  char v77; // [rsp+63Ch] [rbp-2F4h]
  char v78; // [rsp+640h] [rbp-2F0h] BYREF
  char *v79; // [rsp+740h] [rbp-1F0h]
  __int64 v80; // [rsp+748h] [rbp-1E8h]
  char v81; // [rsp+750h] [rbp-1E0h] BYREF
  _QWORD *v82; // [rsp+7D0h] [rbp-160h]
  __int64 v83; // [rsp+7D8h] [rbp-158h]
  __int64 v84; // [rsp+7E0h] [rbp-150h]
  __int64 v85; // [rsp+7E8h] [rbp-148h]
  int v86; // [rsp+7F0h] [rbp-140h]
  __int64 v87; // [rsp+7F8h] [rbp-138h]
  __int64 v88; // [rsp+800h] [rbp-130h]
  __int64 v89; // [rsp+808h] [rbp-128h]
  int v90; // [rsp+810h] [rbp-120h]
  __int64 v91; // [rsp+830h] [rbp-100h]
  __int64 v92; // [rsp+850h] [rbp-E0h]
  __int64 v93; // [rsp+858h] [rbp-D8h]
  __int64 v94; // [rsp+860h] [rbp-D0h]
  int v95; // [rsp+868h] [rbp-C8h]
  __int64 v96; // [rsp+870h] [rbp-C0h]
  __int64 v97; // [rsp+878h] [rbp-B8h]
  __int64 v98; // [rsp+880h] [rbp-B0h]
  int v99; // [rsp+888h] [rbp-A8h]
  __int64 v100; // [rsp+890h] [rbp-A0h]
  __int64 v101; // [rsp+898h] [rbp-98h]
  __int64 v102; // [rsp+8A0h] [rbp-90h]
  int v103; // [rsp+8B0h] [rbp-80h]
  __int64 v104; // [rsp+8B8h] [rbp-78h]
  __int64 v105; // [rsp+8C0h] [rbp-70h]
  __int64 v106; // [rsp+8C8h] [rbp-68h]
  int v107; // [rsp+8D0h] [rbp-60h]
  char v108; // [rsp+8D8h] [rbp-58h]
  char *v109; // [rsp+8E0h] [rbp-50h]
  __int64 v110; // [rsp+8E8h] [rbp-48h]
  char v111; // [rsp+8F0h] [rbp-40h] BYREF

  v2 = *(__int64 **)(a1 + 40);
  v5[0] = a2;
  v5[1] = v2;
  sub_A558A0((__int64)v6, (__int64)v2, 1);
  v7 = v2 + 29;
  v8 = v2 + 39;
  v3 = *v2;
  v11 = 1;
  v9 = v3;
  v10 = 0;
  v12 = &v14;
  v13 = 0x100000000LL;
  v15 = &v17;
  v16 = 0x600000000LL;
  v23 = &v27;
  v29 = &v33;
  v39 = &v43;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v24 = 16;
  v25 = 0;
  v26 = 1;
  v28 = 0;
  v30 = 32;
  v31 = 0;
  v32 = 1;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v40 = 2;
  v41 = 0;
  v42 = 1;
  v45 = 0;
  v54 = v56;
  v59 = &v63;
  v68 = &v72;
  v64 = &v66;
  v74 = &v78;
  v65 = 0x400000000LL;
  v44 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v55 = 0;
  memset(v56, 0, sizeof(v56));
  v57 = 0;
  v58 = 0;
  v60 = 32;
  v61 = 0;
  v62 = 1;
  v67 = 0;
  v69 = 32;
  v70 = 0;
  v71 = 1;
  v73 = 0;
  v75 = 32;
  v76 = 0;
  v77 = 1;
  v79 = &v81;
  v80 = 0x1000000000LL;
  v82 = v5;
  v109 = &v111;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 2;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v110 = 0x400000000LL;
  LODWORD(v2) = sub_C05FA0(v5, a1);
  sub_C08A70((__int64)v5, a1);
  return (unsigned int)v2 ^ 1;
}
