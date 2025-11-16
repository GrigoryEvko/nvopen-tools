// Function: sub_36E4430
// Address: 0x36e4430
//
void __fastcall sub_36E4430(__int64 a1, __int64 a2, char a3, __m128i a4)
{
  int v6; // r13d
  __int64 v7; // rax
  int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r13
  int v17; // ecx
  __int64 v18; // rax
  _QWORD *v19; // rsi
  unsigned __int8 *v20; // r14
  int v21; // edx
  int v22; // r13d
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // r11
  __int64 v28; // rsi
  __int64 v29; // rdx
  int v30; // ecx
  __int64 v31; // r8
  __int64 v32; // rsi
  unsigned __int64 v33; // rcx
  __int64 v34; // r13
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  int v38; // [rsp+4h] [rbp-14Ch]
  __int64 v39; // [rsp+10h] [rbp-140h]
  unsigned __int64 v40; // [rsp+10h] [rbp-140h]
  int v41; // [rsp+18h] [rbp-138h]
  __int64 v42; // [rsp+18h] [rbp-138h]
  __int64 v43; // [rsp+20h] [rbp-130h]
  _QWORD *v44; // [rsp+20h] [rbp-130h]
  int v45; // [rsp+28h] [rbp-128h]
  __int64 v46; // [rsp+38h] [rbp-118h]
  int v47; // [rsp+40h] [rbp-110h]
  __int64 v48; // [rsp+44h] [rbp-10Ch]
  int v49; // [rsp+4Ch] [rbp-104h]
  __int64 v50; // [rsp+50h] [rbp-100h]
  int v51; // [rsp+58h] [rbp-F8h]
  __int64 v52; // [rsp+5Ch] [rbp-F4h]
  int v53; // [rsp+64h] [rbp-ECh]
  __int64 v54; // [rsp+68h] [rbp-E8h]
  int v55; // [rsp+70h] [rbp-E0h]
  __int64 v56; // [rsp+74h] [rbp-DCh]
  int v57; // [rsp+7Ch] [rbp-D4h]
  __int64 v58; // [rsp+80h] [rbp-D0h]
  int v59; // [rsp+88h] [rbp-C8h]
  __int64 v60; // [rsp+8Ch] [rbp-C4h]
  int v61; // [rsp+94h] [rbp-BCh]
  __int64 v62; // [rsp+98h] [rbp-B8h]
  int v63; // [rsp+A0h] [rbp-B0h]
  __int64 v64; // [rsp+A4h] [rbp-ACh]
  int v65; // [rsp+ACh] [rbp-A4h]
  __int64 v66; // [rsp+B0h] [rbp-A0h] BYREF
  int v67; // [rsp+B8h] [rbp-98h]
  __int64 v68; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v69; // [rsp+C8h] [rbp-88h]
  __int64 v70; // [rsp+D0h] [rbp-80h]
  int v71; // [rsp+D8h] [rbp-78h]
  __int64 v72; // [rsp+E0h] [rbp-70h]
  int v73; // [rsp+E8h] [rbp-68h]
  __int64 v74; // [rsp+F0h] [rbp-60h]
  int v75; // [rsp+F8h] [rbp-58h]
  __int64 v76; // [rsp+100h] [rbp-50h]
  int v77; // [rsp+108h] [rbp-48h]
  __int64 v78; // [rsp+110h] [rbp-40h]
  int v79; // [rsp+118h] [rbp-38h]

  v6 = *(unsigned __int16 *)(a2 + 96);
  v7 = *(_QWORD *)(a2 + 104);
  LOWORD(v68) = v6;
  v69 = v7;
  if ( (_WORD)v6 )
  {
    if ( (unsigned __int16)(v6 - 17) <= 0xD3u )
      LOWORD(v6) = word_4456580[v6 - 1];
  }
  else if ( sub_30070B0((__int64)&v68) )
  {
    LOWORD(v6) = sub_3009970((__int64)&v68, a2, v11, v12, v13);
  }
  v8 = *(_DWORD *)(a2 + 24);
  if ( a3 )
  {
    v47 = 2698;
    v9 = v8 - 562;
    v46 = 0xA8900000A88LL;
    v48 = 0xA8000000A7FLL;
    v50 = 0xA8300000A82LL;
    v52 = 0xA8600000A85LL;
    v54 = 0xA7A00000A79LL;
    v49 = 2689;
    v51 = 2692;
    v53 = 2695;
    v55 = 2683;
    v56 = 0xA7D00000A7CLL;
    v57 = 2686;
    switch ( (__int16)v6 )
    {
      case 5:
        v38 = *((_DWORD *)&v46 + v9);
        goto LABEL_10;
      case 6:
        v38 = *((_DWORD *)&v48 + v9);
        goto LABEL_10;
      case 7:
        v38 = *((_DWORD *)&v50 + v9);
        goto LABEL_10;
      case 8:
        v38 = *((_DWORD *)&v52 + v9);
        goto LABEL_10;
      case 12:
        v38 = *((_DWORD *)&v54 + v9);
        goto LABEL_10;
      case 13:
        v38 = *((_DWORD *)&v56 + v9);
        goto LABEL_10;
      default:
        goto LABEL_32;
    }
  }
  v10 = v8 - 556;
  v58 = 0xA9B00000A9ALL;
  v60 = 0xA9200000A91LL;
  v62 = 0xA9500000A94LL;
  v59 = 2716;
  v61 = 2707;
  v63 = 2710;
  v64 = 0xA9800000A97LL;
  v65 = 2713;
  v66 = 0xA8C00000A8BLL;
  v67 = 2701;
  v68 = 0xA8F00000A8ELL;
  LODWORD(v69) = 2704;
  switch ( (__int16)v6 )
  {
    case 5:
      v38 = *((_DWORD *)&v58 + v10);
      break;
    case 6:
      v38 = *((_DWORD *)&v60 + v10);
      break;
    case 7:
      v38 = *((_DWORD *)&v62 + v10);
      break;
    case 8:
      v38 = *((_DWORD *)&v64 + v10);
      break;
    case 12:
      v38 = *((_DWORD *)&v66 + v10);
      break;
    case 13:
      v38 = *((_DWORD *)&v68 + v10);
      break;
    default:
LABEL_32:
      BUG();
  }
LABEL_10:
  v14 = *(__int64 **)(a2 + 40);
  v15 = *(_QWORD *)(a2 + 80);
  v16 = *(_QWORD *)(a1 + 64);
  v43 = *v14;
  v17 = *((_DWORD *)v14 + 2);
  v68 = v15;
  v41 = v17;
  if ( v15 )
  {
    sub_B96E90((__int64)&v68, v15, 1);
    v14 = *(__int64 **)(a2 + 40);
  }
  LODWORD(v69) = *(_DWORD *)(a2 + 72);
  v18 = *(_QWORD *)(v14[5] + 96);
  v19 = *(_QWORD **)(v18 + 24);
  if ( *(_DWORD *)(v18 + 32) > 0x40u )
    v19 = (_QWORD *)*v19;
  v20 = sub_3400BD0(v16, (__int64)v19, (__int64)&v68, 8, 0, 1u, a4, 0);
  v22 = v21;
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
  v23 = *(_QWORD *)(a2 + 40);
  v24 = *(_QWORD *)(v23 + 120);
  v45 = *(_DWORD *)(v23 + 88);
  v25 = *(_QWORD *)(v23 + 128);
  v39 = *(_QWORD *)(v23 + 80);
  v66 = 0;
  v67 = 0;
  v68 = 0;
  LODWORD(v69) = 0;
  sub_36DF750(a1, v24, v25, (__int64)&v66, (__int64)&v68, a4);
  v26 = *(_QWORD *)(a2 + 40);
  v27 = *(_QWORD **)(a1 + 64);
  v28 = v68;
  v29 = *(_QWORD *)(v26 + 160);
  LODWORD(v26) = *(_DWORD *)(v26 + 168);
  v68 = (__int64)v20;
  v30 = v69;
  v31 = *(unsigned int *)(a2 + 68);
  v70 = v39;
  v77 = v26;
  v74 = v28;
  v71 = v45;
  v32 = *(_QWORD *)(a2 + 80);
  v78 = v43;
  v72 = v66;
  LODWORD(v69) = v22;
  v75 = v30;
  v73 = v67;
  v33 = *(_QWORD *)(a2 + 48);
  v76 = v29;
  v79 = v41;
  v66 = v32;
  if ( v32 )
  {
    v40 = v33;
    v42 = v31;
    v44 = v27;
    sub_B96E90((__int64)&v66, v32, 1);
    v33 = v40;
    v31 = v42;
    v27 = v44;
  }
  v67 = *(_DWORD *)(a2 + 72);
  v34 = sub_33E66D0(v27, v38, (__int64)&v66, v33, v31, (__int64)&v66, (unsigned __int64 *)&v68, 6);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v34, v35, v36, v37);
  sub_3421DB0(v34);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v66 )
    sub_B91220((__int64)&v66, v66);
}
