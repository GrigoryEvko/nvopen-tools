// Function: sub_2AB9880
// Address: 0x2ab9880
//
void __fastcall sub_2AB9880(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rdi
  unsigned int v7; // eax
  char v8; // si
  __int64 v9; // rax
  __int64 v10; // rdi
  int v11; // r13d
  int v12; // ecx
  __int64 *v13; // r13
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  unsigned __int8 *v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // r13
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rdx
  int v32; // ecx
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // eax
  char v36; // al
  _BYTE *v37; // r13
  __int64 v38; // rax
  int v39; // edx
  __int64 v40; // rbx
  int v41; // eax
  _BYTE *v42; // r13
  int v43; // ecx
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 *v46; // [rsp+8h] [rbp-188h]
  __int64 v47; // [rsp+10h] [rbp-180h]
  __int64 v48; // [rsp+10h] [rbp-180h]
  _BYTE *v49; // [rsp+10h] [rbp-180h]
  _BYTE *v51; // [rsp+38h] [rbp-158h]
  int v52; // [rsp+54h] [rbp-13Ch]
  __int64 v53; // [rsp+58h] [rbp-138h]
  unsigned int v54; // [rsp+68h] [rbp-128h]
  _DWORD v55[8]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v56; // [rsp+90h] [rbp-100h]
  void *v57[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v58; // [rsp+C0h] [rbp-D0h]
  unsigned int *v59[2]; // [rsp+D0h] [rbp-C0h] BYREF
  _BYTE v60[32]; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+100h] [rbp-90h]
  __int64 v62; // [rsp+108h] [rbp-88h]
  __int16 v63; // [rsp+110h] [rbp-80h]
  __int64 *v64; // [rsp+118h] [rbp-78h]
  void **v65; // [rsp+120h] [rbp-70h]
  void **v66; // [rsp+128h] [rbp-68h]
  __int64 v67; // [rsp+130h] [rbp-60h]
  int v68; // [rsp+138h] [rbp-58h]
  __int16 v69; // [rsp+13Ch] [rbp-54h]
  char v70; // [rsp+13Eh] [rbp-52h]
  __int64 v71; // [rsp+140h] [rbp-50h]
  __int64 v72; // [rsp+148h] [rbp-48h]
  void *v73; // [rsp+150h] [rbp-40h] BYREF
  void *v74; // [rsp+158h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 240);
  v4 = *(_QWORD *)(a1 + 360);
  v5 = sub_986580(v3);
  v70 = 7;
  v64 = (__int64 *)sub_BD5C60(v5);
  v65 = &v73;
  v66 = &v74;
  v63 = 0;
  v59[0] = (unsigned int *)v60;
  v73 = &unk_49DA100;
  v59[1] = (unsigned int *)0x200000000LL;
  v67 = 0;
  v74 = &unk_49DA0B0;
  v68 = 0;
  v69 = 512;
  v71 = 0;
  v72 = 0;
  v61 = 0;
  v62 = 0;
  sub_D5F1F0((__int64)v59, v5);
  v6 = *(_QWORD *)(a1 + 384);
  v7 = *(_DWORD *)(a1 + 72);
  if ( !*(_BYTE *)(a1 + 76) || (v8 = 1, !v7) )
    v8 = v7 > 1;
  v52 = 36 - (((unsigned __int8)sub_2AB31C0(v6, v8) == 0) - 1);
  v47 = *(_QWORD *)(v4 + 8);
  v9 = sub_ACD720(v64);
  v10 = *(_QWORD *)(a1 + 384);
  v53 = v9;
  if ( *(_BYTE *)(v10 + 108) && (v11 = *(_DWORD *)(v10 + 100)) != 0 )
  {
    if ( *(_BYTE *)(a1 + 76) )
    {
      v36 = sub_2AA9E60((__int64 *)v10, *(_QWORD *)(a1 + 72), *(_DWORD *)(a1 + 88), 1);
      if ( v11 != 4 && v36 != 1 )
      {
        sub_BCB300((__int64)v57, v47);
        v37 = (_BYTE *)sub_AD8D80(v47, (__int64)v57);
        sub_969240((__int64 *)v57);
        v58 = 257;
        v38 = sub_929DE0(v59, v37, (_BYTE *)v4, (__int64)v57, 0, 0);
        v39 = *(_DWORD *)(a1 + 72);
        v40 = v38;
        v41 = *(_DWORD *)(a1 + 88);
        v56 = 257;
        if ( (unsigned int)(v41 * v39) < *(_DWORD *)(a1 + 80) )
        {
          v42 = (_BYTE *)sub_2AB26E0((__int64)v59, v47, *(_QWORD *)(a1 + 80), 1);
          if ( *(_BYTE *)(a1 + 76) )
          {
            v43 = *(_DWORD *)(a1 + 88);
            v44 = *(_QWORD *)(a1 + 72);
            v58 = 257;
            v45 = sub_2AB26E0((__int64)v59, v47, v44, v43);
            v42 = (_BYTE *)sub_B33C40((__int64)v59, 0x16Du, (__int64)v42, v45, v54, (__int64)v57);
          }
        }
        else
        {
          v42 = (_BYTE *)sub_2AB26E0((__int64)v59, v47, *(_QWORD *)(a1 + 72), v41);
        }
        v53 = sub_92B530(v59, 0x24u, v40, v42, (__int64)v55);
      }
    }
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 88);
    if ( (unsigned int)(v12 * *(_DWORD *)(a1 + 72)) < *(_DWORD *)(a1 + 80) )
    {
      v51 = (_BYTE *)sub_2AB26E0((__int64)v59, v47, *(_QWORD *)(a1 + 80), 1);
      if ( *(_BYTE *)(a1 + 76) )
      {
        v32 = *(_DWORD *)(a1 + 88);
        v33 = *(_QWORD *)(a1 + 72);
        v58 = 257;
        v55[1] = 0;
        v34 = sub_2AB26E0((__int64)v59, v47, v33, v32);
        v51 = (_BYTE *)sub_B33C40((__int64)v59, 0x16Du, (__int64)v51, v34, v55[0], (__int64)v57);
      }
    }
    else
    {
      v51 = (_BYTE *)sub_2AB26E0((__int64)v59, v47, *(_QWORD *)(a1 + 72), v12);
    }
    v13 = *(__int64 **)(*(_QWORD *)(a1 + 16) + 112LL);
    v48 = *(_QWORD *)(a1 + 8);
    v14 = sub_DD8400((__int64)v13, v4);
    v49 = (_BYTE *)sub_DE4F70(v13, (__int64)v14, v48);
    v15 = sub_DD8400((__int64)v13, (__int64)v51);
    if ( (unsigned __int8)sub_DC3A60((__int64)v13, v52 & 0x7F, v49, v15) )
    {
      v53 = sub_ACD6D0(v64);
    }
    else
    {
      v46 = sub_DD8400((__int64)v13, (__int64)v51);
      v35 = sub_B52870(v52);
      if ( !(unsigned __int8)sub_DC3A60((__int64)v13, v35, v49, v46) )
      {
        v57[0] = "min.iters.check";
        v58 = 259;
        v53 = sub_92B530(v59, v52, v4, v51, (__int64)v57);
      }
    }
  }
  v16 = *(_QWORD *)(a1 + 24);
  v17 = *(_QWORD *)(a1 + 32);
  v57[0] = "vector.ph";
  v58 = 259;
  v18 = sub_986580(v3);
  v19 = sub_F36960(v3, (__int64 *)(v18 + 24), 0, v17, v16, 0, v57, 0);
  *(_QWORD *)(a1 + 240) = v19;
  v20 = v19;
  v21 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v23 = v21;
  if ( v21 )
    sub_B4C9A0((__int64)v21, a2, v20, v53, 3u, v22, 0, 0);
  v24 = sub_D47930(*(_QWORD *)(a1 + 8));
  v25 = sub_986580(v24);
  if ( (unsigned __int8)sub_BC8700(v25) )
    sub_BC8EC0((__int64)v23, (unsigned int *)&unk_439F0B8, 2, 0);
  v26 = sub_986580(v3);
  sub_F34910(v26, v23);
  v29 = *(unsigned int *)(a1 + 272);
  v30 = *(unsigned int *)(a1 + 276);
  if ( v29 + 1 > v30 )
  {
    sub_C8D5F0(a1 + 264, (const void *)(a1 + 280), v29 + 1, 8u, v27, v28);
    v29 = *(unsigned int *)(a1 + 272);
  }
  v31 = *(_QWORD *)(a1 + 264);
  *(_QWORD *)(v31 + 8 * v29) = v3;
  ++*(_DWORD *)(a1 + 272);
  sub_2AB95C0(a1, v3, v31, v30, v27, v28);
  nullsub_61();
  v73 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v59[0] != v60 )
    _libc_free((unsigned __int64)v59[0]);
}
