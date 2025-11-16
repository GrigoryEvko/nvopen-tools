// Function: sub_2463FC0
// Address: 0x2463fc0
//
unsigned __int64 __fastcall sub_2463FC0(__int64 a1, __int64 a2, unsigned int **a3, unsigned __int16 a4)
{
  __int64 v6; // rdx
  int v7; // ecx
  int v8; // eax
  __int64 v9; // rdx
  int v10; // ecx
  int v11; // eax
  __int64 v12; // rdx
  int v13; // ecx
  __int64 *v14; // rax
  __int64 *v15; // rdi
  __int64 *v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rax
  int v19; // ecx
  __int64 v20; // rdx
  int v21; // ecx
  int v22; // eax
  __int64 v23; // rdx
  int v24; // ecx
  __int64 *v25; // rax
  __int64 *v26; // rdi
  __int64 *v27; // rdi
  __int64 v28; // r8
  _BYTE *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  _BYTE *v32; // r13
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  unsigned __int64 v36; // rdx
  int v37; // ecx
  bool v38; // zf
  __int64 v39; // rcx
  int v40; // esi
  int v41; // eax
  __int64 v42; // rcx
  int v43; // esi
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 *v46; // rdi
  __int64 v47; // rax
  __int64 *v48; // rdi
  __int64 **v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // edx
  __int64 *v53; // rax
  __int64 **v54; // rcx
  unsigned __int8 *v56; // rax
  _BYTE *v57; // rdx
  __int64 v58; // rdx
  _BYTE *v59; // rax
  unsigned __int8 *v60; // rax
  _BYTE *v61; // rdx
  _BYTE *v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  _BYTE *v65; // rax
  __int64 v67; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v68; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v69; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v70; // [rsp+8h] [rbp-D8h]
  __int64 v71; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v72; // [rsp+8h] [rbp-D8h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  __int64 v74; // [rsp+20h] [rbp-C0h]
  __int64 v75; // [rsp+28h] [rbp-B8h]
  __int64 v76; // [rsp+30h] [rbp-B0h]
  __int64 v77; // [rsp+38h] [rbp-A8h]
  __int64 v78; // [rsp+40h] [rbp-A0h]
  __int64 v79; // [rsp+48h] [rbp-98h]
  __int64 v80; // [rsp+50h] [rbp-90h]
  __int64 v81; // [rsp+58h] [rbp-88h]
  __int64 v82; // [rsp+60h] [rbp-80h]
  __int64 v83; // [rsp+68h] [rbp-78h]
  int v84; // [rsp+70h] [rbp-70h]
  __int64 v85; // [rsp+78h] [rbp-68h]
  _BYTE v86[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v87; // [rsp+A0h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v7 - 17) > 1 )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL);
    v28 = v17;
  }
  else
  {
    v8 = *(_DWORD *)(v6 + 32);
    v9 = *(_QWORD *)(v6 + 24);
    BYTE4(v73) = (_BYTE)v7 == 18;
    LODWORD(v73) = v8;
    v10 = *(unsigned __int8 *)(v9 + 8);
    if ( (unsigned int)(v10 - 17) > 1 )
    {
      v16 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 80LL);
    }
    else
    {
      v11 = *(_DWORD *)(v9 + 32);
      v12 = *(_QWORD *)(v9 + 24);
      BYTE4(v74) = (_BYTE)v10 == 18;
      LODWORD(v74) = v11;
      v13 = *(unsigned __int8 *)(v12 + 8);
      if ( (unsigned int)(v13 - 17) > 1 )
      {
        v15 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 80LL);
      }
      else
      {
        BYTE4(v75) = (_BYTE)v13 == 18;
        LODWORD(v75) = *(_DWORD *)(v12 + 32);
        v14 = (__int64 *)sub_2462D00(a1, *(_QWORD *)(v12 + 24));
        v15 = (__int64 *)sub_BCE1B0(v14, v75);
      }
      v16 = (__int64 *)sub_BCE1B0(v15, v74);
    }
    v17 = sub_BCE1B0(v16, v73);
    v18 = *(_QWORD *)(a2 + 8);
    v19 = *(unsigned __int8 *)(v18 + 8);
    if ( (unsigned int)(v19 - 17) > 1 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL);
    }
    else
    {
      BYTE4(v76) = (_BYTE)v19 == 18;
      LODWORD(v76) = *(_DWORD *)(v18 + 32);
      v20 = *(_QWORD *)(v18 + 24);
      v21 = *(unsigned __int8 *)(v20 + 8);
      if ( (unsigned int)(v21 - 17) > 1 )
      {
        v27 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 80LL);
      }
      else
      {
        v22 = *(_DWORD *)(v20 + 32);
        v23 = *(_QWORD *)(v20 + 24);
        BYTE4(v77) = (_BYTE)v21 == 18;
        LODWORD(v77) = v22;
        v24 = *(unsigned __int8 *)(v23 + 8);
        if ( (unsigned int)(v24 - 17) > 1 )
        {
          v26 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 80LL);
        }
        else
        {
          BYTE4(v78) = (_BYTE)v24 == 18;
          LODWORD(v78) = *(_DWORD *)(v23 + 32);
          v25 = (__int64 *)sub_2462D00(a1, *(_QWORD *)(v23 + 24));
          v26 = (__int64 *)sub_BCE1B0(v25, v78);
        }
        v27 = (__int64 *)sub_BCE1B0(v26, v77);
      }
      v28 = sub_BCE1B0(v27, v76);
    }
  }
  v67 = v28;
  v87 = 257;
  v29 = sub_94BCF0(a3, a2, v28, (__int64)v86);
  v30 = *(_QWORD *)(a1 + 8);
  v31 = v67;
  v32 = v29;
  v33 = *(__int64 **)(v30 + 688);
  v34 = *v33;
  if ( *v33 )
  {
    v87 = 257;
    v62 = (_BYTE *)sub_2462030(a1, v67, ~v34);
    v63 = sub_A82350(a3, v32, v62, (__int64)v86);
    v30 = *(_QWORD *)(a1 + 8);
    v31 = v67;
    v32 = (_BYTE *)v63;
    v33 = *(__int64 **)(v30 + 688);
  }
  v35 = v33[1];
  if ( v35 )
  {
    v87 = 257;
    if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
    {
      v61 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v30 + 80), v35, 0);
    }
    else
    {
      v71 = v31;
      v60 = (unsigned __int8 *)sub_2462030(a1, *(_QWORD *)(v31 + 24), v35);
      BYTE4(v79) = *(_BYTE *)(v71 + 8) == 18;
      LODWORD(v79) = *(_DWORD *)(v71 + 32);
      v61 = (_BYTE *)sub_AD5E10(v79, v60);
    }
    v32 = (_BYTE *)sub_A825B0(a3, v32, v61, (__int64)v86);
    v33 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 688LL);
  }
  v36 = (unsigned __int64)v32;
  if ( v33[2] )
  {
    v58 = v33[2];
    v87 = 257;
    v59 = (_BYTE *)sub_2462030(a1, v17, v58);
    v36 = sub_929C50(a3, v32, v59, (__int64)v86, 0, 0);
  }
  v87 = 257;
  v37 = *(unsigned __int8 *)(v17 + 8);
  if ( (unsigned int)(v37 - 17) > 1 )
  {
    v49 = *(__int64 ***)(*(_QWORD *)(a1 + 8) + 96LL);
  }
  else
  {
    v38 = (_BYTE)v37 == 18;
    v39 = *(_QWORD *)(v17 + 24);
    BYTE4(v80) = v38;
    LODWORD(v80) = *(_DWORD *)(v17 + 32);
    v40 = *(unsigned __int8 *)(v39 + 8);
    if ( (unsigned int)(v40 - 17) > 1 )
    {
      v48 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL);
    }
    else
    {
      v41 = *(_DWORD *)(v39 + 32);
      v42 = *(_QWORD *)(v39 + 24);
      BYTE4(v81) = (_BYTE)v40 == 18;
      LODWORD(v81) = v41;
      v43 = *(unsigned __int8 *)(v42 + 8);
      if ( (unsigned int)(v43 - 17) > 1 )
      {
        v46 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL);
      }
      else
      {
        v68 = v36;
        BYTE4(v82) = (_BYTE)v43 == 18;
        LODWORD(v82) = *(_DWORD *)(v42 + 32);
        v44 = (__int64 *)sub_2462BA0(a1, *(_QWORD *)(v42 + 24));
        v45 = sub_BCE1B0(v44, v82);
        v36 = v68;
        v46 = (__int64 *)v45;
      }
      v69 = v36;
      v47 = sub_BCE1B0(v46, v81);
      v36 = v69;
      v48 = (__int64 *)v47;
    }
    v72 = v36;
    v64 = sub_BCE1B0(v48, v80);
    v36 = v72;
    v49 = (__int64 **)v64;
  }
  v70 = sub_24633A0((__int64 *)a3, 0x30u, v36, v49, (__int64)v86, 0, v85, 0);
  v50 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v50 + 4) )
  {
    v51 = *(_QWORD *)(*(_QWORD *)(v50 + 688) + 24LL);
    if ( v51 )
    {
      v87 = 257;
      v65 = (_BYTE *)sub_2462030(a1, v17, v51);
      v32 = (_BYTE *)sub_929C50(a3, v32, v65, (__int64)v86, 0, 0);
    }
    if ( !HIBYTE(a4) || (unsigned __int8)byte_4FE8EA9 > (unsigned __int8)a4 )
    {
      v87 = 257;
      if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 > 1 )
      {
        v57 = (_BYTE *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL), -1LL << byte_4FE8EA9, 0);
      }
      else
      {
        v56 = (unsigned __int8 *)sub_2462030(a1, *(_QWORD *)(v17 + 24), -1LL << byte_4FE8EA9);
        BYTE4(v83) = *(_BYTE *)(v17 + 8) == 18;
        LODWORD(v83) = *(_DWORD *)(v17 + 32);
        v57 = (_BYTE *)sub_AD5E10(v83, v56);
      }
      v32 = (_BYTE *)sub_A82350(a3, v32, v57, (__int64)v86);
    }
    v87 = 257;
    v52 = *(unsigned __int8 *)(v17 + 8);
    if ( (unsigned int)(v52 - 17) > 1 )
    {
      v54 = *(__int64 ***)(*(_QWORD *)(a1 + 8) + 96LL);
    }
    else
    {
      BYTE4(v85) = (_BYTE)v52 == 18;
      LODWORD(v85) = *(_DWORD *)(v17 + 32);
      v53 = (__int64 *)sub_2462BA0(a1, *(_QWORD *)(v17 + 24));
      v54 = (__int64 **)sub_BCE1B0(v53, v85);
    }
    sub_24633A0((__int64 *)a3, 0x30u, (unsigned __int64)v32, v54, (__int64)v86, 0, v84, 0);
  }
  return v70;
}
