// Function: sub_1805380
// Address: 0x1805380
//
unsigned __int64 __fastcall sub_1805380(_QWORD *a1, __int64 a2)
{
  _QWORD *v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  int v13; // edx
  __int64 **v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 result; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rax
  __int64 *v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 **v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 **v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r13
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 *v34; // r15
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rdx
  unsigned __int8 *v39; // rsi
  __int64 v40; // rax
  __int64 *v41; // r15
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // rdx
  unsigned __int8 *v46; // rsi
  unsigned __int8 *v47; // [rsp+18h] [rbp-148h] BYREF
  _BYTE v48[16]; // [rsp+20h] [rbp-140h] BYREF
  __int16 v49; // [rsp+30h] [rbp-130h]
  __int64 v50[2]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v51; // [rsp+50h] [rbp-110h]
  __int64 v52[2]; // [rsp+60h] [rbp-100h] BYREF
  __int16 v53; // [rsp+70h] [rbp-F0h]
  __int64 v54; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+88h] [rbp-D8h]
  __int64 v56; // [rsp+90h] [rbp-D0h]
  char v57[16]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v58; // [rsp+B0h] [rbp-B0h]
  unsigned __int8 *v59[2]; // [rsp+C0h] [rbp-A0h] BYREF
  __int16 v60; // [rsp+D0h] [rbp-90h]
  unsigned __int8 *v61; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v62; // [rsp+E8h] [rbp-78h]
  __int64 *v63; // [rsp+F0h] [rbp-70h]
  _QWORD *v64; // [rsp+F8h] [rbp-68h]
  __int64 v65; // [rsp+100h] [rbp-60h]
  int v66; // [rsp+108h] [rbp-58h]
  __int64 v67; // [rsp+110h] [rbp-50h]
  __int64 v68; // [rsp+118h] [rbp-48h]

  v4 = (_QWORD *)sub_16498A0(a2);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v61 = 0;
  v64 = v4;
  v6 = *(_QWORD *)(a2 + 40);
  v65 = 0;
  v62 = v6;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v63 = (__int64 *)(a2 + 24);
  v59[0] = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)v59, (__int64)v5, 2);
    if ( v61 )
      sub_161E7C0((__int64)&v61, (__int64)v61);
    v61 = v59[0];
    if ( v59[0] )
      sub_1623210((__int64)v59, v59[0], (__int64)&v61);
  }
  v7 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v7 + 16) )
    BUG();
  v8 = *(_DWORD *)(v7 + 36);
  if ( (v8 & 0xFFFFFFFD) == 0x85 )
  {
    v58 = 257;
    v49 = 257;
    v9 = sub_16471D0(v64, 0);
    v10 = sub_12A95D0((__int64 *)&v61, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v9, (__int64)v48);
    v51 = 257;
    v54 = v10;
    v11 = sub_16471D0(v64, 0);
    v12 = sub_12A95D0(
            (__int64 *)&v61,
            *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
            v11,
            (__int64)v50);
    v13 = *(_DWORD *)(a2 + 20);
    v14 = (__int64 **)a1[29];
    v55 = v12;
    v53 = 257;
    v15 = v13 & 0xFFFFFFF;
    v16 = *(_QWORD *)(a2 + 24 * (2 - v15));
    if ( v14 != *(__int64 ***)v16 )
    {
      if ( *(_BYTE *)(v16 + 16) > 0x10u )
      {
        v20 = *(_QWORD **)(a2 + 24 * (2 - v15));
        v60 = 257;
        v21 = sub_15FE0A0(v20, (__int64)v14, 0, (__int64)v59, 0);
        v16 = v21;
        if ( v62 )
        {
          v22 = v63;
          sub_157E9D0(v62 + 40, v21);
          v23 = *(_QWORD *)(v16 + 24);
          v24 = *v22;
          *(_QWORD *)(v16 + 32) = v22;
          v24 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v16 + 24) = v24 | v23 & 7;
          *(_QWORD *)(v24 + 8) = v16 + 24;
          *v22 = *v22 & 7 | (v16 + 24);
        }
        sub_164B780(v16, v52);
        sub_12A86E0((__int64 *)&v61, v16);
      }
      else
      {
        v16 = sub_15A4750(*(__int64 ****)(a2 + 24 * (2 - v15)), v14, 0);
      }
    }
    v17 = *(_QWORD *)(a2 - 24);
    v56 = v16;
    if ( *(_BYTE *)(v17 + 16) )
      BUG();
    v18 = a1[87];
    if ( *(_DWORD *)(v17 + 36) == 135 )
      v18 = a1[86];
LABEL_14:
    sub_1285290((__int64 *)&v61, *(_QWORD *)(v18 + 24), v18, (int)&v54, 3, (__int64)v57, 0);
    goto LABEL_16;
  }
  if ( v8 == 137 )
  {
    v58 = 257;
    v49 = 257;
    v25 = sub_16471D0(v64, 0);
    v54 = sub_12A95D0((__int64 *)&v61, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v25, (__int64)v48);
    v51 = 257;
    v26 = (__int64 **)sub_1643350(v64);
    v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v28 = *(_QWORD *)(a2 + 24 * (1 - v27));
    if ( v26 != *(__int64 ***)v28 )
    {
      if ( *(_BYTE *)(v28 + 16) > 0x10u )
      {
        v60 = 257;
        v40 = sub_15FE0A0((_QWORD *)v28, (__int64)v26, 0, (__int64)v59, 0);
        v28 = v40;
        if ( v62 )
        {
          v41 = v63;
          sub_157E9D0(v62 + 40, v40);
          v42 = *(_QWORD *)(v28 + 24);
          v43 = *v41;
          *(_QWORD *)(v28 + 32) = v41;
          v43 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v28 + 24) = v43 | v42 & 7;
          *(_QWORD *)(v43 + 8) = v28 + 24;
          *v41 = *v41 & 7 | (v28 + 24);
        }
        sub_164B780(v28, v50);
        if ( v61 )
        {
          v52[0] = (__int64)v61;
          sub_1623A60((__int64)v52, (__int64)v61, 2);
          v44 = *(_QWORD *)(v28 + 48);
          v45 = v28 + 48;
          if ( v44 )
          {
            sub_161E7C0(v28 + 48, v44);
            v45 = v28 + 48;
          }
          v46 = (unsigned __int8 *)v52[0];
          *(_QWORD *)(v28 + 48) = v52[0];
          if ( v46 )
            sub_1623210((__int64)v52, v46, v45);
        }
        v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      }
      else
      {
        v28 = sub_15A4750(*(__int64 ****)(a2 + 24 * (1 - v27)), v26, 0);
        v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      }
    }
    v55 = v28;
    v53 = 257;
    v29 = (__int64 **)a1[29];
    v30 = 3 * (2 - v27);
    v31 = *(_QWORD *)(a2 + 8 * v30);
    if ( v29 != *(__int64 ***)v31 )
    {
      if ( *(_BYTE *)(v31 + 16) > 0x10u )
      {
        v32 = *(_QWORD **)(a2 + 8 * v30);
        v60 = 257;
        v33 = sub_15FE0A0(v32, (__int64)v29, 0, (__int64)v59, 0);
        v31 = v33;
        if ( v62 )
        {
          v34 = v63;
          sub_157E9D0(v62 + 40, v33);
          v35 = *(_QWORD *)(v31 + 24);
          v36 = *v34;
          *(_QWORD *)(v31 + 32) = v34;
          v36 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v31 + 24) = v36 | v35 & 7;
          *(_QWORD *)(v36 + 8) = v31 + 24;
          *v34 = *v34 & 7 | (v31 + 24);
        }
        sub_164B780(v31, v52);
        if ( v61 )
        {
          v47 = v61;
          sub_1623A60((__int64)&v47, (__int64)v61, 2);
          v37 = *(_QWORD *)(v31 + 48);
          v38 = v31 + 48;
          if ( v37 )
          {
            sub_161E7C0(v31 + 48, v37);
            v38 = v31 + 48;
          }
          v39 = v47;
          *(_QWORD *)(v31 + 48) = v47;
          if ( v39 )
            sub_1623210((__int64)&v47, v39, v38);
        }
      }
      else
      {
        v31 = sub_15A4750(*(__int64 ****)(a2 + 8 * v30), v29, 0);
      }
    }
    v56 = v31;
    v18 = a1[88];
    goto LABEL_14;
  }
LABEL_16:
  result = sub_15F20C0((_QWORD *)a2);
  if ( v61 )
    return sub_161E7C0((__int64)&v61, (__int64)v61);
  return result;
}
