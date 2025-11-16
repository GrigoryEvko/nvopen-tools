// Function: sub_3137750
// Address: 0x3137750
//
__int64 __fastcall sub_3137750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v7; // rbx
  _QWORD *v8; // r15
  __int64 v10; // rdx
  __int16 v11; // ax
  _BYTE *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 *v18; // rdx
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  int v23; // edx
  unsigned __int64 v24; // rax
  bool v25; // cf
  _QWORD *v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // r14
  __int64 v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // rbx
  __int64 v40; // rdx
  unsigned int v41; // esi
  _QWORD *v42; // r13
  unsigned __int64 v43; // rsi
  int v44; // eax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int16 v47; // dx
  char v48; // cl
  char v49; // dl
  __int64 *v50; // [rsp+0h] [rbp-A0h]
  __int64 v51; // [rsp+0h] [rbp-A0h]
  __int64 v52; // [rsp+18h] [rbp-88h]
  __int64 v53; // [rsp+18h] [rbp-88h]
  _QWORD *v54; // [rsp+18h] [rbp-88h]
  __int64 v55; // [rsp+20h] [rbp-80h]
  _QWORD *v56; // [rsp+20h] [rbp-80h]
  __int64 v57; // [rsp+28h] [rbp-78h]
  const char *v59; // [rsp+40h] [rbp-60h] BYREF
  __int64 v60; // [rsp+48h] [rbp-58h]
  __int16 v61; // [rsp+60h] [rbp-40h]

  v7 = a2;
  v8 = *(_QWORD **)(a2 + 560);
  if ( a6 == 1 && a4 )
  {
    v61 = 257;
    v57 = a2 + 512;
    v13 = (_BYTE *)sub_AD6530(*(_QWORD *)(a4 + 8), a2);
    v14 = sub_92B530((unsigned int **)(a2 + 512), 0x21u, a4, v13, (__int64)&v59);
    v61 = 259;
    v59 = "omp_region.body";
    v55 = **(_QWORD **)(a2 + 504);
    v15 = sub_22077B0(0x50u);
    v16 = v15;
    if ( v15 )
      sub_AA4D50(v15, v55, (__int64)&v59, 0, 0);
    v52 = *(_QWORD *)(a2 + 584);
    sub_B43C20((__int64)&v59, v16);
    v56 = sub_BD2C40(72, unk_3F148B8);
    if ( v56 )
      sub_B4C8A0((__int64)v56, v52, (__int64)v59, v60);
    v17 = v8[9];
    v18 = (__int64 *)v8[4];
    v19 = v8 + 6;
    v50 = v18;
    v53 = v17;
    sub_B2B790(v17 + 72, v16);
    v20 = *(_QWORD *)(v16 + 24) & 7LL;
    v21 = *v50;
    *(_QWORD *)(v16 + 32) = v50;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v16 + 24) = v21 | v20;
    *(_QWORD *)(v21 + 8) = v16 + 24;
    *v50 = *v50 & 7 | (v16 + 24);
    sub_AA4C30(v16, *(_BYTE *)(v53 + 128));
    v22 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v22 == v19 )
    {
      v54 = 0;
    }
    else
    {
      if ( !v22 )
        BUG();
      v23 = *(unsigned __int8 *)(v22 - 24);
      v24 = v22 - 24;
      v25 = (unsigned int)(v23 - 30) < 0xB;
      v26 = 0;
      if ( v25 )
        v26 = (_QWORD *)v24;
      v54 = v26;
    }
    v61 = 257;
    v27 = sub_BD2C40(72, 3u);
    v28 = (__int64)v27;
    if ( v27 )
      sub_B4C9A0((__int64)v27, v16, a5, v14, 3u, 0, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(v7 + 600) + 16LL))(
      *(_QWORD *)(v7 + 600),
      v28,
      &v59,
      *(_QWORD *)(v57 + 56),
      *(_QWORD *)(v57 + 64));
    v29 = *(_QWORD *)(v7 + 512);
    v30 = 16LL * *(unsigned int *)(v7 + 520);
    if ( v29 != v29 + v30 )
    {
      v51 = v7;
      v31 = v29 + v30;
      v32 = *(_QWORD *)(v7 + 512);
      do
      {
        v33 = *(_QWORD *)(v32 + 8);
        v34 = *(_DWORD *)v32;
        v32 += 16;
        sub_B99FD0(v28, v34, v33);
      }
      while ( v31 != v32 );
      v7 = v51;
    }
    sub_B43D10(v54);
    sub_D5F1F0(v57, (__int64)v56);
    v35 = *(_QWORD *)(v57 + 56);
    v36 = *(_QWORD *)(v57 + 64);
    v37 = *(_QWORD *)(v7 + 600);
    v61 = 257;
    (*(void (__fastcall **)(__int64, _QWORD *, const char **, __int64, __int64))(*(_QWORD *)v37 + 16LL))(
      v37,
      v54,
      &v59,
      v35,
      v36);
    v38 = *(_QWORD *)(v7 + 512);
    v39 = v38 + 16LL * *(unsigned int *)(v7 + 520);
    while ( v39 != v38 )
    {
      v40 = *(_QWORD *)(v38 + 8);
      v41 = *(_DWORD *)v38;
      v38 += 16;
      sub_B99FD0((__int64)v54, v41, v40);
    }
    v42 = (_QWORD *)(v16 + 48);
    sub_B43D60(v56);
    v43 = *v42 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v43 == v42 )
    {
      v45 = 0;
    }
    else
    {
      if ( !v43 )
        BUG();
      v44 = *(unsigned __int8 *)(v43 - 24);
      v45 = v43 - 24;
      if ( (unsigned int)(v44 - 30) >= 0xB )
        v45 = 0;
    }
    sub_D5F1F0(v57, v45);
    v46 = sub_AA5190(a5);
    if ( v46 )
    {
      v48 = v47;
      v49 = HIBYTE(v47);
    }
    else
    {
      v49 = 0;
      v48 = 0;
    }
    *(_QWORD *)(a1 + 8) = v46;
    *(_BYTE *)(a1 + 16) = v48;
    *(_QWORD *)a1 = a5;
    *(_BYTE *)(a1 + 17) = v49;
  }
  else
  {
    v10 = *(_QWORD *)(a2 + 568);
    v11 = *(_WORD *)(a2 + 576);
    *(_QWORD *)a1 = v8;
    *(_QWORD *)(a1 + 8) = v10;
    *(_WORD *)(a1 + 16) = v11;
  }
  return a1;
}
