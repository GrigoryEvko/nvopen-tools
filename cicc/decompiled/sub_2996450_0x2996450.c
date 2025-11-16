// Function: sub_2996450
// Address: 0x2996450
//
__int64 __fastcall sub_2996450(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // rdx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned int v10; // r12d
  __int64 v12[10]; // [rsp+0h] [rbp-3C0h] BYREF
  char v13; // [rsp+50h] [rbp-370h] BYREF
  __int64 v14; // [rsp+90h] [rbp-330h]
  char *v15; // [rsp+98h] [rbp-328h]
  __int64 v16; // [rsp+A0h] [rbp-320h]
  int v17; // [rsp+A8h] [rbp-318h]
  char v18; // [rsp+ACh] [rbp-314h]
  char v19; // [rsp+B0h] [rbp-310h] BYREF
  __int64 v20; // [rsp+F0h] [rbp-2D0h]
  char *v21; // [rsp+F8h] [rbp-2C8h]
  __int64 v22; // [rsp+100h] [rbp-2C0h]
  int v23; // [rsp+108h] [rbp-2B8h]
  char v24; // [rsp+10Ch] [rbp-2B4h]
  char v25; // [rsp+110h] [rbp-2B0h] BYREF
  char *v26; // [rsp+150h] [rbp-270h]
  __int64 v27; // [rsp+158h] [rbp-268h]
  char v28; // [rsp+160h] [rbp-260h] BYREF
  __int64 v29; // [rsp+220h] [rbp-1A0h]
  __int64 v30; // [rsp+228h] [rbp-198h]
  __int64 v31; // [rsp+230h] [rbp-190h]
  int v32; // [rsp+238h] [rbp-188h]
  __int64 v33; // [rsp+240h] [rbp-180h]
  __int64 v34; // [rsp+248h] [rbp-178h]
  __int64 v35; // [rsp+250h] [rbp-170h]
  int v36; // [rsp+258h] [rbp-168h]
  _QWORD *v37; // [rsp+260h] [rbp-160h]
  __int64 v38; // [rsp+268h] [rbp-158h]
  _QWORD v39[3]; // [rsp+270h] [rbp-150h] BYREF
  int v40; // [rsp+288h] [rbp-138h]
  char *v41; // [rsp+290h] [rbp-130h]
  __int64 v42; // [rsp+298h] [rbp-128h]
  char v43; // [rsp+2A0h] [rbp-120h] BYREF
  __int64 v44; // [rsp+2E0h] [rbp-E0h]
  __int64 v45; // [rsp+2E8h] [rbp-D8h]
  __int64 v46; // [rsp+2F0h] [rbp-D0h]
  int v47; // [rsp+2F8h] [rbp-C8h]
  __int64 v48; // [rsp+300h] [rbp-C0h]
  __int64 v49; // [rsp+308h] [rbp-B8h]
  __int64 v50; // [rsp+310h] [rbp-B0h]
  int v51; // [rsp+318h] [rbp-A8h]
  char *v52; // [rsp+320h] [rbp-A0h]
  __int64 v53; // [rsp+328h] [rbp-98h]
  char v54; // [rsp+330h] [rbp-90h] BYREF
  __int64 v55; // [rsp+370h] [rbp-50h]
  __int64 v56; // [rsp+378h] [rbp-48h]
  __int64 v57; // [rsp+380h] [rbp-40h]
  int v58; // [rsp+388h] [rbp-38h]

  v15 = &v19;
  v21 = &v25;
  v26 = &v28;
  v12[8] = (__int64)&v13;
  v37 = v39;
  v12[9] = 0x800000000LL;
  v27 = 0x800000000LL;
  v12[6] = 0;
  v14 = 0;
  v16 = 8;
  v17 = 0;
  v18 = 1;
  v20 = 0;
  v22 = 8;
  v23 = 0;
  v24 = 1;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v38 = 0;
  memset(v39, 0, sizeof(v39));
  v41 = &v43;
  v40 = 0;
  v42 = 0x800000000LL;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = &v54;
  v53 = 0x800000000LL;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  sub_2988360(v12, a2);
  if ( *(_BYTE *)(a1 + 169) )
  {
    v2 = *(__int64 **)(a1 + 8);
    v3 = *v2;
    v4 = v2[1];
    if ( v3 == v4 )
LABEL_19:
      BUG();
    while ( *(_UNKNOWN **)v3 != &unk_4F8FC84 )
    {
      v3 += 16;
      if ( v4 == v3 )
        goto LABEL_19;
    }
    v5 = (__int64 *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                       *(_QWORD *)(v3 + 8),
                       &unk_4F8FC84)
                   + 184);
    if ( a2[4] )
    {
      if ( (unsigned __int8)sub_298D780((__int64)v12, a2, v5) )
        goto LABEL_15;
    }
  }
  v6 = *(__int64 **)(a1 + 8);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_18:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F8144C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_18;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F8144C)
     + 176;
  if ( !a2[4] )
LABEL_15:
    v10 = 0;
  else
    v10 = sub_2994AD0((__int64)v12, (__int64)a2, v9);
  sub_2989910((__int64)v12);
  return v10;
}
