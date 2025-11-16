// Function: sub_2AB9F30
// Address: 0x2ab9f30
//
unsigned __int8 *__fastcall sub_2AB9F30(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // r13
  __int64 v6; // rbx
  unsigned __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned int v11; // edx
  char v12; // si
  char v13; // al
  __int64 v14; // rsi
  unsigned int v15; // r14d
  _QWORD *v16; // r15
  __int64 v17; // r8
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int8 *v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned int v32; // eax
  __int64 v33; // rbx
  __int64 v34; // r14
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rbx
  _BYTE *v41; // r14
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // [rsp+0h] [rbp-170h]
  __int64 v45; // [rsp+0h] [rbp-170h]
  int v47; // [rsp+28h] [rbp-148h]
  __int64 v48; // [rsp+28h] [rbp-148h]
  __int64 v50; // [rsp+30h] [rbp-140h]
  __int64 v51; // [rsp+40h] [rbp-130h]
  __int64 v52; // [rsp+48h] [rbp-128h]
  const char *v53; // [rsp+50h] [rbp-120h] BYREF
  char v54; // [rsp+70h] [rbp-100h]
  char v55; // [rsp+71h] [rbp-FFh]
  void *v56[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v57; // [rsp+A0h] [rbp-D0h]
  _BYTE *v58; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+B8h] [rbp-B8h]
  _BYTE v60[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+E0h] [rbp-90h]
  __int64 v62; // [rsp+E8h] [rbp-88h]
  __int64 v63; // [rsp+F0h] [rbp-80h]
  __int64 v64; // [rsp+F8h] [rbp-78h]
  void **v65; // [rsp+100h] [rbp-70h]
  void **v66; // [rsp+108h] [rbp-68h]
  __int64 v67; // [rsp+110h] [rbp-60h]
  int v68; // [rsp+118h] [rbp-58h]
  __int16 v69; // [rsp+11Ch] [rbp-54h]
  char v70; // [rsp+11Eh] [rbp-52h]
  __int64 v71; // [rsp+120h] [rbp-50h]
  __int64 v72; // [rsp+128h] [rbp-48h]
  void *v73; // [rsp+130h] [rbp-40h] BYREF
  void *v74; // [rsp+138h] [rbp-38h] BYREF

  if ( a3 )
  {
    v4 = *(_QWORD *)(a1 + 480);
    v51 = *(_QWORD *)(v4 + 12);
    v47 = *(_DWORD *)(v4 + 20);
  }
  else
  {
    v51 = *(_QWORD *)(a1 + 72);
    v47 = *(_DWORD *)(a1 + 88);
  }
  v5 = *(unsigned __int8 **)(a1 + 240);
  v6 = *(_QWORD *)(a1 + 360);
  v7 = sub_986580((__int64)v5);
  v8 = sub_BD5C60(v7);
  v70 = 7;
  v64 = v8;
  v65 = &v73;
  v66 = &v74;
  LOWORD(v63) = 0;
  v58 = v60;
  v73 = &unk_49DA100;
  v59 = 0x200000000LL;
  v67 = 0;
  v68 = 0;
  v69 = 512;
  v71 = 0;
  v72 = 0;
  v61 = 0;
  v62 = 0;
  v74 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v58, v7);
  v9 = *(_QWORD *)(a1 + 384);
  if ( a3 )
  {
    v10 = *(_QWORD *)(a1 + 480);
    v11 = *(_DWORD *)(v10 + 12);
    if ( !*(_BYTE *)(v10 + 16) || (v12 = 1, !v11) )
      v12 = v11 > 1;
  }
  else
  {
    v32 = *(_DWORD *)(a1 + 72);
    if ( !*(_BYTE *)(a1 + 76) || (v12 = 1, !v32) )
      v12 = v32 > 1;
  }
  v13 = sub_2AB31C0(v9, v12);
  v55 = 1;
  v54 = 3;
  v14 = *(_QWORD *)(v6 + 8);
  v15 = 36 - ((v13 == 0) - 1);
  v53 = "min.iters.check";
  v44 = sub_2AB26E0((__int64)&v58, v14, v51, v47);
  v16 = (_QWORD *)(*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v65 + 7))(v65, v15, v6, v44);
  if ( !v16 )
  {
    v57 = 257;
    v16 = sub_BD2C40(72, unk_3F10FD0);
    if ( v16 )
    {
      v37 = *(_QWORD *)(v6 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 > 1 )
      {
        v39 = sub_BCB2A0(*(_QWORD **)v37);
      }
      else
      {
        BYTE4(v52) = *(_BYTE *)(v37 + 8) == 18;
        LODWORD(v52) = *(_DWORD *)(v37 + 32);
        v38 = (__int64 *)sub_BCB2A0(*(_QWORD **)v37);
        v39 = sub_BCE1B0(v38, v52);
      }
      sub_B523C0((__int64)v16, v39, 53, v15, v6, v44, (__int64)v56, 0, 0, 0);
    }
    (*((void (__fastcall **)(void **, _QWORD *, const char **, __int64, __int64))*v66 + 2))(v66, v16, &v53, v62, v63);
    if ( v58 != &v58[16 * (unsigned int)v59] )
    {
      v45 = v6;
      v40 = (unsigned __int64)v58;
      v41 = &v58[16 * (unsigned int)v59];
      do
      {
        v42 = *(_QWORD *)(v40 + 8);
        v43 = *(_DWORD *)v40;
        v40 += 16LL;
        sub_B99FD0((__int64)v16, v43, v42);
      }
      while ( v41 != (_BYTE *)v40 );
      v6 = v45;
    }
  }
  HIBYTE(v57) = 1;
  if ( a3 )
  {
    v17 = *(_QWORD *)(a1 + 24);
    v18 = *(_QWORD *)(a1 + 32);
    v56[0] = "vector.ph";
    v50 = v17;
    LOBYTE(v57) = 3;
    v19 = sub_986580((__int64)v5);
    *(_QWORD *)(a1 + 240) = sub_F36960((__int64)v5, (__int64 *)(v19 + 24), 0, v18, v50, 0, v56, 0);
    sub_B1A4E0(a1 + 264, (__int64)v5);
    *(_QWORD *)(*(_QWORD *)(a1 + 480) + 56LL) = v6;
    v20 = *(_QWORD *)(a1 + 240);
  }
  else
  {
    LOBYTE(v57) = 3;
    v56[0] = "vector.main.loop.iter.check";
    sub_BD6B50(v5, (const char **)v56);
    v33 = *(_QWORD *)(a1 + 24);
    v34 = *(_QWORD *)(a1 + 32);
    v56[0] = "vector.ph";
    v57 = 259;
    v35 = sub_986580((__int64)v5);
    v36 = sub_F36960((__int64)v5, (__int64 *)(v35 + 24), 0, v34, v33, 0, v56, 0);
    *(_QWORD *)(a1 + 240) = v36;
    v20 = v36;
  }
  v48 = v20;
  v21 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v23 = v21;
  if ( v21 )
    sub_B4C9A0((__int64)v21, a2, v48, (__int64)v16, 3u, v22, 0, 0);
  v24 = sub_D47930(*(_QWORD *)(a1 + 8));
  v25 = sub_986580(v24);
  if ( (unsigned __int8)sub_BC8700(v25) )
    sub_BC8EC0((__int64)v23, (unsigned int *)&unk_439F0B8, 2, 0);
  v26 = sub_986580((__int64)v5);
  sub_F34910(v26, v23);
  sub_2AB95C0(a1, (__int64)v5, v27, v28, v29, v30);
  nullsub_61();
  v73 = &unk_49DA100;
  nullsub_63();
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  return v5;
}
