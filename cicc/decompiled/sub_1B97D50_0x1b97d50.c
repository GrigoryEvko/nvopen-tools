// Function: sub_1B97D50
// Address: 0x1b97d50
//
__int64 __fastcall sub_1B97D50(
        __int64 a1,
        __int64 *a2,
        int a3,
        _QWORD *a4,
        unsigned int a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v9; // r13
  bool v10; // zf
  int v11; // r14d
  int v12; // r15d
  int v13; // r8d
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 *v17; // rdi
  __int64 *v18; // r15
  __int64 v19; // rax
  __int64 *v20; // r9
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v27; // rcx
  __int64 v28; // rdx
  int v29; // r13d
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 *v32; // r13
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r15
  __int64 v37; // r14
  __int64 v38; // r8
  int v39; // r9d
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 *v42; // rdi
  __int64 v43; // r13
  __int64 *v44; // r15
  __int64 v45; // rax
  __int64 v46; // rsi
  bool v47; // cc
  __int64 v48; // rax
  __int64 *v49; // r13
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // [rsp+0h] [rbp-F0h]
  __int64 v53; // [rsp+8h] [rbp-E8h]
  __int64 v54; // [rsp+8h] [rbp-E8h]
  __int64 v58; // [rsp+28h] [rbp-C8h]
  __int64 v59; // [rsp+28h] [rbp-C8h]
  __int64 v60; // [rsp+28h] [rbp-C8h]
  __int64 v61; // [rsp+28h] [rbp-C8h]
  __int64 v62; // [rsp+28h] [rbp-C8h]
  __int64 v63[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v64; // [rsp+40h] [rbp-B0h]
  __int64 v65[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v66; // [rsp+60h] [rbp-90h]
  __int64 *v67; // [rsp+70h] [rbp-80h] BYREF
  __int64 v68; // [rsp+78h] [rbp-78h]
  _BYTE v69[112]; // [rsp+80h] [rbp-70h] BYREF

  v9 = *a2;
  v58 = *(_QWORD *)(*a2 + 32);
  if ( *(_BYTE *)(*a2 + 8) == 16 )
    v9 = **(_QWORD **)(v9 + 16);
  v68 = 0x800000000LL;
  v10 = *(_BYTE *)(v9 + 8) == 11;
  v67 = (__int64 *)v69;
  if ( !v10 )
  {
    if ( (int)v58 <= 0 )
    {
      v16 = 0;
      v17 = (__int64 *)v69;
    }
    else
    {
      v11 = a3 + v58;
      v12 = a3;
      do
      {
        a6 = (double)v12;
        v14 = sub_15A10B0(v9, (double)v12);
        v15 = (unsigned int)v68;
        if ( (unsigned int)v68 >= HIDWORD(v68) )
        {
          v52 = v14;
          sub_16CD150((__int64)&v67, v69, 0, 8, v13, v14);
          v15 = (unsigned int)v68;
          v14 = v52;
        }
        ++v12;
        v67[v15] = v14;
        v16 = (unsigned int)(v68 + 1);
        LODWORD(v68) = v68 + 1;
      }
      while ( v11 != v12 );
      v17 = v67;
    }
    v53 = sub_15A01B0(v17, v16);
    v66 = 257;
    v18 = (__int64 *)(a1 + 96);
    v19 = sub_156DA60((__int64 *)(a1 + 96), v58, a4, v65);
    v20 = (__int64 *)v53;
    v64 = 257;
    v21 = v19;
    if ( *(_BYTE *)(v53 + 16) <= 0x10u
      && *(_BYTE *)(v19 + 16) <= 0x10u
      && (v22 = sub_15A2A30((__int64 *)0x10, (__int64 *)v53, v19, 0, 0, a6, a7, a8),
          v20 = (__int64 *)v53,
          (v23 = v22) != 0) )
    {
      if ( *(_BYTE *)(v22 + 16) <= 0x17u )
        goto LABEL_14;
    }
    else
    {
      v66 = 257;
      v27 = sub_15FB440(16, v20, v21, (__int64)v65, 0);
      v28 = *(_QWORD *)(a1 + 128);
      v29 = *(_DWORD *)(a1 + 136);
      if ( v28 )
      {
        v59 = v27;
        sub_1625C10(v27, 3, v28);
        v27 = v59;
      }
      v60 = v27;
      sub_15F2440(v27, v29);
      v30 = v60;
      v31 = *(_QWORD *)(a1 + 104);
      if ( v31 )
      {
        v32 = *(__int64 **)(a1 + 112);
        sub_157E9D0(v31 + 40, v60);
        v30 = v60;
        v33 = *v32;
        v34 = *(_QWORD *)(v60 + 24);
        *(_QWORD *)(v60 + 32) = v32;
        v33 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v60 + 24) = v33 | v34 & 7;
        *(_QWORD *)(v33 + 8) = v60 + 24;
        *v32 = *v32 & 7 | (v60 + 24);
      }
      v61 = v30;
      sub_164B780(v30, v63);
      sub_12A86E0(v18, v61);
      v23 = v61;
      if ( *(_BYTE *)(v61 + 16) <= 0x17u )
        goto LABEL_14;
    }
    v62 = v23;
    sub_15F2440(v23, -1);
    v23 = v62;
LABEL_14:
    v65[0] = (__int64)"induction";
    v66 = 259;
    v24 = sub_1904E90((__int64)v18, a5, (__int64)a2, v23, v65, 0, a6, a7, a8);
    v25 = v24;
    if ( *(_BYTE *)(v24 + 16) > 0x17u )
      sub_15F2440(v24, -1);
    goto LABEL_16;
  }
  if ( (int)v58 <= 0 )
  {
    v41 = 0;
    v42 = (__int64 *)v69;
  }
  else
  {
    v35 = a3;
    v36 = a3 + 1LL;
    v37 = v36 + (unsigned int)(v58 - 1);
    while ( 1 )
    {
      v38 = sub_15A0680(v9, v35, 0);
      v40 = (unsigned int)v68;
      if ( (unsigned int)v68 >= HIDWORD(v68) )
      {
        v54 = v38;
        sub_16CD150((__int64)&v67, v69, 0, 8, v38, v39);
        v40 = (unsigned int)v68;
        v38 = v54;
      }
      v67[v40] = v38;
      v35 = v36;
      v41 = (unsigned int)(v68 + 1);
      LODWORD(v68) = v68 + 1;
      if ( v37 == v36 )
        break;
      ++v36;
    }
    v42 = v67;
  }
  v43 = sub_15A01B0(v42, v41);
  v66 = 257;
  v44 = (__int64 *)(a1 + 96);
  v45 = sub_156DA60((__int64 *)(a1 + 96), v58, a4, v65);
  v66 = 257;
  if ( *(_BYTE *)(v43 + 16) > 0x10u || *(_BYTE *)(v45 + 16) > 0x10u )
    v46 = (__int64)sub_17D2EF0(v44, 15, (__int64 *)v43, v45, v65, 0, 0);
  else
    v46 = sub_15A2C20((__int64 *)v43, v45, 0, 0, a6, a7, a8);
  v47 = *((_BYTE *)a2 + 16) <= 0x10u;
  v63[0] = (__int64)"induction";
  v64 = 259;
  if ( v47 && *(_BYTE *)(v46 + 16) <= 0x10u )
  {
    v25 = sub_15A2B30(a2, v46, 0, 0, a6, a7, a8);
  }
  else
  {
    v66 = 257;
    v25 = sub_15FB440(11, a2, v46, (__int64)v65, 0);
    v48 = *(_QWORD *)(a1 + 104);
    if ( v48 )
    {
      v49 = *(__int64 **)(a1 + 112);
      sub_157E9D0(v48 + 40, v25);
      v50 = *(_QWORD *)(v25 + 24);
      v51 = *v49;
      *(_QWORD *)(v25 + 32) = v49;
      v51 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v25 + 24) = v51 | v50 & 7;
      *(_QWORD *)(v51 + 8) = v25 + 24;
      *v49 = *v49 & 7 | (v25 + 24);
    }
    sub_164B780(v25, v63);
    sub_12A86E0(v44, v25);
  }
LABEL_16:
  if ( v67 != (__int64 *)v69 )
    _libc_free((unsigned __int64)v67);
  return v25;
}
