// Function: sub_17D5930
// Address: 0x17d5930
//
__int64 *__fastcall sub_17D5930(__int64 a1, char a2, double a3, double a4, double a5)
{
  __int64 *result; // rax
  __int64 v7; // r12
  _QWORD *v8; // rax
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 *v14; // rdi
  unsigned int v15; // ebx
  _BYTE *v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // r13
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 *v23; // rsi
  unsigned int v24; // eax
  __int16 v25; // cx
  __int128 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // eax
  unsigned int v30; // r13d
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rsi
  __int64 *v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // eax
  unsigned int v39; // ecx
  char v40; // al
  unsigned int v41; // ecx
  unsigned int v42; // eax
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rax
  _BYTE *v53; // [rsp+10h] [rbp-150h]
  __int64 *v54; // [rsp+18h] [rbp-148h]
  __int64 v55; // [rsp+20h] [rbp-140h]
  unsigned __int64 *v56; // [rsp+20h] [rbp-140h]
  __int64 *v57; // [rsp+20h] [rbp-140h]
  __int64 v58; // [rsp+28h] [rbp-138h]
  __int64 *v59; // [rsp+30h] [rbp-130h]
  unsigned __int64 v60; // [rsp+30h] [rbp-130h]
  __int64 v61; // [rsp+40h] [rbp-120h]
  __int64 *v62; // [rsp+48h] [rbp-118h]
  _QWORD v63[2]; // [rsp+50h] [rbp-110h] BYREF
  __int16 v64; // [rsp+60h] [rbp-100h]
  _QWORD v65[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v66; // [rsp+80h] [rbp-E0h]
  unsigned __int8 *v67; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v68; // [rsp+98h] [rbp-C8h]
  unsigned __int64 *v69; // [rsp+A0h] [rbp-C0h]
  _QWORD *v70; // [rsp+A8h] [rbp-B8h]
  __int64 v71; // [rsp+B0h] [rbp-B0h]
  int v72; // [rsp+B8h] [rbp-A8h]
  __int64 v73; // [rsp+C0h] [rbp-A0h]
  __int64 v74; // [rsp+C8h] [rbp-98h]
  unsigned __int8 *v75[2]; // [rsp+E0h] [rbp-80h] BYREF
  __int16 v76; // [rsp+F0h] [rbp-70h]

  result = *(__int64 **)(a1 + 896);
  v62 = result;
  v54 = &result[*(unsigned int *)(a1 + 904)];
  if ( result != v54 )
  {
    do
    {
      v7 = *v62;
      v8 = (_QWORD *)sub_16498A0(*v62);
      v73 = 0;
      v74 = 0;
      v9 = *(unsigned __int8 **)(v7 + 48);
      v70 = v8;
      v72 = 0;
      v10 = *(_QWORD *)(v7 + 40);
      v67 = 0;
      v68 = v10;
      v71 = 0;
      v69 = (unsigned __int64 *)(v7 + 24);
      v75[0] = v9;
      if ( v9 )
      {
        sub_1623A60((__int64)v75, (__int64)v9, 2);
        if ( v67 )
          sub_161E7C0((__int64)&v67, (__int64)v67);
        v67 = v75[0];
        if ( v75[0] )
          sub_1623210((__int64)v75, v75[0], (__int64)&v67);
      }
      v59 = *(__int64 **)(v7 - 48);
      v58 = *(_QWORD *)(v7 - 24);
      if ( sub_15F32D0(v7) )
      {
        v11 = *v59;
        v14 = sub_17CD8D0((_QWORD *)a1, *v59);
        if ( !v14 )
          BUG();
        v61 = sub_15A06D0((__int64 **)v14, v11, v12, v13);
      }
      else
      {
        *((_QWORD *)&v26 + 1) = v59;
        *(_QWORD *)&v26 = a1;
        v61 = (__int64)sub_17D4DA0(v26);
      }
      v15 = 1 << (*(unsigned __int16 *)(v7 + 18) >> 1) >> 1;
      v55 = sub_17CFB40(a1, v58, (__int64 *)&v67, *(__int64 **)v61, v15);
      v53 = v16;
      v76 = 257;
      v17 = sub_1648A60(64, 2u);
      v18 = v17;
      if ( v17 )
        sub_15F9650((__int64)v17, v61, v55, 0, 0);
      if ( v68 )
      {
        v56 = v69;
        sub_157E9D0(v68 + 40, (__int64)v18);
        v19 = *v56;
        v20 = v18[3] & 7LL;
        v18[4] = v56;
        v19 &= 0xFFFFFFFFFFFFFFF8LL;
        v18[3] = v19 | v20;
        *(_QWORD *)(v19 + 8) = v18 + 3;
        *v56 = *v56 & 7 | (unsigned __int64)(v18 + 3);
      }
      sub_164B780((__int64)v18, (__int64 *)v75);
      if ( v67 )
      {
        v65[0] = v67;
        sub_1623A60((__int64)v65, (__int64)v67, 2);
        v21 = v18[6];
        v22 = (__int64)(v18 + 6);
        if ( v21 )
        {
          sub_161E7C0((__int64)(v18 + 6), v21);
          v22 = (__int64)(v18 + 6);
        }
        v23 = (unsigned __int8 *)v65[0];
        v18[6] = v65[0];
        if ( v23 )
          sub_1623210((__int64)v65, v23, v22);
      }
      sub_15F9450((__int64)v18, v15);
      if ( sub_15F32D0(v7) )
      {
        v24 = *(unsigned __int16 *)(v7 + 18);
        switch ( (v24 >> 7) & 7 )
        {
          case 0u:
            v25 = 0;
            break;
          case 1u:
          case 2u:
          case 5u:
            v25 = 640;
            break;
          case 3u:
          case 7u:
            v25 = 896;
            break;
          case 4u:
          case 6u:
            v25 = 768;
            break;
        }
        *(_WORD *)(v7 + 18) = *(_WORD *)(v7 + 18) & 0x8000 | v25 | v24 & 0x7C7F;
      }
      if ( !*(_DWORD *)(*(_QWORD *)(a1 + 8) + 156LL) || sub_15F32D0(v7) )
        goto LABEL_24;
      v29 = 4;
      if ( v15 >= 4 )
        v29 = v15;
      v30 = v29;
      v31 = sub_17D4880(a1, (const char *)v59, v27, v28);
      v32 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
      v33 = *(_QWORD *)v61;
      v60 = (unsigned __int64)(sub_127FA20(v32, *(_QWORD *)v61) + 7) >> 3;
      v34 = *(__int64 **)v61;
      v35 = *(unsigned __int8 *)(*(_QWORD *)v61 + 8LL);
      if ( (unsigned int)(v35 - 13) <= 1 )
        goto LABEL_36;
      if ( (_BYTE)v35 == 16 )
      {
        v57 = *(__int64 **)v61;
        v33 = *((_DWORD *)v57 + 8) * (unsigned int)sub_1643030(v34[3]);
        v37 = sub_1644900(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 168LL), v33);
        v34 = v57;
        v35 = v37;
        if ( v57 != (__int64 *)v37 )
        {
          v33 = 47;
          v76 = 257;
          v61 = sub_12AA3B0((__int64 *)&v67, 0x2Fu, v61, v37, (__int64)v75);
          if ( !v61 )
            BUG();
        }
      }
      if ( *(_BYTE *)(v61 + 16) > 0x10u )
      {
        v38 = sub_127FA20(v32, *(_QWORD *)v61);
        if ( v38 <= 8 )
        {
          v40 = a2;
          v41 = 0;
        }
        else
        {
          v39 = (v38 + 7) >> 3;
          v40 = a2;
          v41 = v39 - 1;
          if ( v41 )
          {
            _BitScanReverse(&v42, v41);
            v41 = 32 - (v42 ^ 0x1F);
            v40 = a2 & (v41 <= 3);
          }
        }
        if ( v40 )
        {
          v43 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * v41 + 296);
          v76 = 257;
          v44 = sub_1644C60(v70, 8 << v41);
          v45 = sub_12AA3B0((__int64 *)&v67, 0x25u, v61, v44, (__int64)v75);
          v76 = 257;
          v64 = 257;
          v65[0] = v45;
          v46 = sub_16471D0(v70, 0);
          v65[1] = sub_12A95D0((__int64 *)&v67, v58, v46, (__int64)v63);
          v66 = v31;
          sub_1285290((__int64 *)&v67, *(_QWORD *)(*(_QWORD *)v43 + 24LL), v43, (int)v65, 3, (__int64)v75, 0);
        }
        else
        {
          v75[0] = "_mscmp";
          v76 = 259;
          v47 = sub_17CDAE0((_QWORD *)a1, *(_QWORD *)v61);
          v48 = sub_12AA0C0((__int64 *)&v67, 0x21u, (_BYTE *)v61, v47, (__int64)v75);
          v49 = v69;
          if ( v69 )
            v49 = v69 - 3;
          v50 = sub_1AA92B0(v48, v49, 0, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 424LL), 0, 0);
          sub_17CE510((__int64)v75, v50, 0, 0, 0);
          v51 = *(_QWORD *)(a1 + 8);
          v63[0] = v31;
          if ( *(int *)(v51 + 156) > 1 )
          {
            LOWORD(v66) = 257;
            v31 = sub_1285290(
                    (__int64 *)v75,
                    *(_QWORD *)(**(_QWORD **)(v51 + 344) + 24LL),
                    *(_QWORD *)(v51 + 344),
                    (int)v63,
                    1,
                    (__int64)v65,
                    0);
          }
          sub_17D3020((_QWORD *)a1, (__int64 *)v75, v31, v53, v60, v30, a3, a4, a5);
          sub_17CD270((__int64 *)v75);
        }
        goto LABEL_24;
      }
      if ( byte_4FA4360 && !sub_1595F50(v61, v33, (__int64)v34, v35) )
      {
LABEL_36:
        v36 = *(_QWORD *)(a1 + 8);
        v65[0] = v31;
        if ( *(int *)(v36 + 156) > 1 )
        {
          v76 = 257;
          v31 = sub_1285290(
                  (__int64 *)&v67,
                  *(_QWORD *)(**(_QWORD **)(v36 + 344) + 24LL),
                  *(_QWORD *)(v36 + 344),
                  (int)v65,
                  1,
                  (__int64)v75,
                  0);
        }
        sub_17D3020((_QWORD *)a1, (__int64 *)&v67, v31, v53, v60, v30, a3, a4, a5);
      }
LABEL_24:
      if ( v67 )
        sub_161E7C0((__int64)&v67, (__int64)v67);
      result = ++v62;
    }
    while ( v54 != v62 );
  }
  return result;
}
