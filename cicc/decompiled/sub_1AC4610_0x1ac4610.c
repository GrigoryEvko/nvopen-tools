// Function: sub_1AC4610
// Address: 0x1ac4610
//
__int64 __fastcall sub_1AC4610(__int64 ***a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // rdi
  __int64 *v9; // r14
  _QWORD *v10; // r13
  __int64 *v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // r15
  __int64 result; // rax
  __int64 v15; // r13
  __int64 v16; // rsi
  __int64 *v17; // r12
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // r10
  __int64 v24; // rsi
  __int64 *v25; // r8
  __int64 **v26; // rax
  __int64 v27; // r15
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  __int64 v30; // rsi
  unsigned __int8 *v31; // rsi
  __int64 v32; // [rsp-10h] [rbp-D0h]
  __int64 v33; // [rsp+8h] [rbp-B8h]
  __int64 v34; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+18h] [rbp-A8h]
  __int64 *v36; // [rsp+18h] [rbp-A8h]
  __int64 *v37; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+20h] [rbp-A0h]
  __int64 *v39; // [rsp+20h] [rbp-A0h]
  __int64 *v40; // [rsp+20h] [rbp-A0h]
  __int64 *v41; // [rsp+20h] [rbp-A0h]
  __int64 *v42; // [rsp+20h] [rbp-A0h]
  __int64 v43; // [rsp+28h] [rbp-98h]
  __int64 v44; // [rsp+30h] [rbp-90h] BYREF
  __int64 v45; // [rsp+38h] [rbp-88h]
  _QWORD v46[2]; // [rsp+40h] [rbp-80h] BYREF
  const char *v47; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v48; // [rsp+58h] [rbp-68h]
  __int16 v49; // [rsp+60h] [rbp-60h]
  const char **v50; // [rsp+70h] [rbp-50h] BYREF
  char *v51; // [rsp+78h] [rbp-48h]
  _QWORD v52[8]; // [rsp+80h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a4 + 40);
  v45 = a3;
  v44 = a2;
  v9 = *(__int64 **)(*(_QWORD *)(v8 + 56) + 40LL);
  v10 = (_QWORD *)sub_157E9C0(v8);
  switch ( v45 )
  {
    case 6LL:
      if ( *(_DWORD *)v44 != 1970234221 || *(_WORD *)(v44 + 4) != 29806 )
        goto LABEL_40;
      goto LABEL_7;
    case 7LL:
      if ( (*(_DWORD *)v44 != 1868786990 || *(_WORD *)(v44 + 4) != 28277 || *(_BYTE *)(v44 + 6) != 116)
        && (*(_DWORD *)v44 != 1868786945 || *(_WORD *)(v44 + 4) != 28277 || *(_BYTE *)(v44 + 6) != 116)
        && (*(_DWORD *)v44 != 1868787039 || *(_WORD *)(v44 + 4) != 28277 || *(_BYTE *)(v44 + 6) != 116) )
      {
        goto LABEL_40;
      }
LABEL_7:
      v11 = (__int64 *)sub_1643270(v10);
      v50 = (const char **)v52;
      v51 = 0;
      v12 = sub_1644EA0(v11, v52, 0, 0);
      v13 = (_QWORD *)sub_1632080((__int64)v9, v44, v45, v12, 0);
      LOWORD(v52[0]) = 257;
      result = (__int64)sub_1648A60(72, 1u);
      v15 = result;
      if ( result )
        result = sub_15F5ED0(result, v13, (__int64)&v50, a4);
      goto LABEL_9;
    case 16LL:
      if ( !(*(_QWORD *)v44 ^ 0x6D5F756E675F5F01LL | *(_QWORD *)(v44 + 8) ^ 0x636E5F746E756F63LL) )
        goto LABEL_7;
      goto LABEL_40;
    case 8LL:
      if ( *(_QWORD *)v44 == 0x746E756F636D5F01LL || *(_QWORD *)v44 == 0x746E756F636D5F5FLL )
        goto LABEL_7;
LABEL_40:
      v47 = "Unknown instrumentation function: '";
      v48 = &v44;
      v50 = &v47;
      v49 = 1283;
      v51 = "'";
      LOWORD(v52[0]) = 770;
      sub_16BCFB0((__int64)&v50, 1u);
    case 29LL:
      if ( *(_QWORD *)v44 ^ 0x72705F6779635F5FLL | *(_QWORD *)(v44 + 8) ^ 0x75665F656C69666FLL
        || *(_QWORD *)(v44 + 16) != 0x7265746E655F636ELL
        || *(_DWORD *)(v44 + 24) != 1918984799
        || *(_BYTE *)(v44 + 28) != 101 )
      {
        goto LABEL_40;
      }
      goto LABEL_7;
    case 24LL:
      if ( *(_QWORD *)v44 ^ 0x72705F6779635F5FLL | *(_QWORD *)(v44 + 8) ^ 0x75665F656C69666FLL
        || *(_QWORD *)(v44 + 16) != 0x7265746E655F636ELL )
      {
        goto LABEL_40;
      }
      break;
    default:
      if ( v45 != 23
        || *(_QWORD *)v44 ^ 0x72705F6779635F5FLL | *(_QWORD *)(v44 + 8) ^ 0x75665F656C69666FLL
        || *(_DWORD *)(v44 + 16) != 1700750190
        || *(_WORD *)(v44 + 20) != 27000
        || *(_BYTE *)(v44 + 22) != 116 )
      {
        goto LABEL_40;
      }
      break;
  }
  v46[0] = sub_16471D0(v10, 0);
  v46[1] = sub_16471D0(v10, 0);
  v18 = (__int64 *)sub_1643270(v10);
  v19 = sub_1644EA0(v18, v46, 2, 0);
  v20 = sub_1632190((__int64)v9, v44, v45, v19);
  LOWORD(v52[0]) = 257;
  v43 = v20;
  v21 = sub_1643350(v10);
  v47 = (const char *)sub_159C470(v21, 0, 0);
  v34 = sub_15E26F0(v9, 186, 0, 0);
  v38 = *(_QWORD *)(*(_QWORD *)v34 + 24LL);
  v22 = sub_1648AB0(72, 2u, 0);
  v23 = v22;
  if ( v22 )
  {
    v33 = v34;
    v35 = (__int64)v22;
    sub_15F1EA0((__int64)v22, **(_QWORD **)(v38 + 16), 54, (__int64)(v22 - 6), 2, a4);
    *(_QWORD *)(v35 + 56) = 0;
    sub_15F5B40(v35, v38, v33, (__int64 *)&v47, 1, (__int64)&v50, 0, 0);
    v23 = (__int64 *)v35;
  }
  v24 = *a5;
  v25 = v23 + 6;
  v50 = (const char **)v24;
  if ( v24 )
  {
    v36 = v23 + 6;
    v39 = v23;
    sub_1623A60((__int64)&v50, v24, 2);
    v25 = v36;
    v23 = v39;
    if ( v36 == (__int64 *)&v50 )
    {
      if ( v50 )
      {
        sub_161E7C0((__int64)&v50, (__int64)v50);
        v23 = v39;
      }
      goto LABEL_32;
    }
    v30 = v39[6];
    if ( !v30 )
    {
LABEL_65:
      v31 = (unsigned __int8 *)v50;
      v23[6] = (__int64)v50;
      if ( v31 )
      {
        v42 = v23;
        sub_1623210((__int64)&v50, v31, (__int64)v25);
        v23 = v42;
      }
      goto LABEL_32;
    }
LABEL_64:
    v37 = v23;
    v41 = v25;
    sub_161E7C0((__int64)v25, v30);
    v23 = v37;
    v25 = v41;
    goto LABEL_65;
  }
  if ( v25 != (__int64 *)&v50 )
  {
    v30 = v23[6];
    if ( v30 )
      goto LABEL_64;
  }
LABEL_32:
  v40 = v23;
  v26 = (__int64 **)sub_16471D0(v10, 0);
  v47 = (const char *)sub_15A4510(a1, v26, 0);
  v48 = v40;
  LOWORD(v52[0]) = 257;
  v27 = *(_QWORD *)(*(_QWORD *)v43 + 24LL);
  result = (__int64)sub_1648AB0(72, 3u, 0);
  v15 = result;
  if ( result )
  {
    sub_15F1EA0(result, **(_QWORD **)(v27 + 16), 54, result - 72, 3, a4);
    *(_QWORD *)(v15 + 56) = 0;
    sub_15F5B40(v15, v27, v43, (__int64 *)&v47, 2, (__int64)&v50, 0, 0);
    result = v32;
  }
LABEL_9:
  v16 = *a5;
  v17 = (__int64 *)(v15 + 48);
  v50 = (const char **)v16;
  if ( v16 )
  {
    result = sub_1623A60((__int64)&v50, v16, 2);
    if ( v17 == (__int64 *)&v50 )
    {
      if ( v50 )
        return sub_161E7C0((__int64)&v50, (__int64)v50);
      return result;
    }
    v28 = *(_QWORD *)(v15 + 48);
    if ( !v28 )
      goto LABEL_37;
  }
  else
  {
    if ( v17 == (__int64 *)&v50 )
      return result;
    v28 = *(_QWORD *)(v15 + 48);
    if ( !v28 )
      return result;
  }
  result = sub_161E7C0(v15 + 48, v28);
LABEL_37:
  v29 = (unsigned __int8 *)v50;
  *(_QWORD *)(v15 + 48) = v50;
  if ( v29 )
    return sub_1623210((__int64)&v50, v29, v15 + 48);
  return result;
}
