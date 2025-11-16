// Function: sub_2ABA4F0
// Address: 0x2aba4f0
//
__int64 __fastcall sub_2ABA4F0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // r14
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // rax
  bool v13; // zf
  unsigned int v14; // eax
  char v15; // si
  char v16; // al
  __int64 v17; // rsi
  unsigned int v18; // r14d
  _QWORD *v19; // r15
  unsigned __int8 *v20; // rax
  __int64 v21; // r9
  unsigned __int8 *v22; // r14
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  unsigned int v25; // eax
  unsigned int v26; // edx
  unsigned __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  _QWORD **v38; // rdx
  int v39; // ecx
  __int64 *v40; // rax
  __int64 v41; // rax
  unsigned int *v42; // r14
  unsigned int *v43; // rbx
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 v46; // [rsp+8h] [rbp-168h]
  __int64 v47; // [rsp+10h] [rbp-160h]
  __int64 v49; // [rsp+48h] [rbp-128h]
  const char *v50; // [rsp+50h] [rbp-120h] BYREF
  char v51; // [rsp+70h] [rbp-100h]
  char v52; // [rsp+71h] [rbp-FFh]
  unsigned int v53[8]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v54; // [rsp+A0h] [rbp-D0h]
  unsigned int *v55; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-B8h]
  _BYTE v57[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+E0h] [rbp-90h]
  __int64 v59; // [rsp+E8h] [rbp-88h]
  __int64 v60; // [rsp+F0h] [rbp-80h]
  __int64 v61; // [rsp+F8h] [rbp-78h]
  void **v62; // [rsp+100h] [rbp-70h]
  void **v63; // [rsp+108h] [rbp-68h]
  __int64 v64; // [rsp+110h] [rbp-60h]
  int v65; // [rsp+118h] [rbp-58h]
  __int16 v66; // [rsp+11Ch] [rbp-54h]
  char v67; // [rsp+11Eh] [rbp-52h]
  __int64 v68; // [rsp+120h] [rbp-50h]
  __int64 v69; // [rsp+128h] [rbp-48h]
  void *v70; // [rsp+130h] [rbp-40h] BYREF
  void *v71; // [rsp+138h] [rbp-38h] BYREF

  v5 = *(_BYTE **)(*(_QWORD *)(a1 + 480) + 56LL);
  v6 = sub_986580(a3);
  v7 = sub_BD5C60(v6);
  v67 = 7;
  v61 = v7;
  v62 = &v70;
  v63 = &v71;
  v56 = 0x200000000LL;
  v66 = 512;
  v70 = &unk_49DA100;
  LOWORD(v60) = 0;
  v55 = (unsigned int *)v57;
  v71 = &unk_49DA0B0;
  v64 = 0;
  v65 = 0;
  v68 = 0;
  v69 = 0;
  v58 = 0;
  v59 = 0;
  sub_D5F1F0((__int64)&v55, v6);
  *(_QWORD *)v53 = "n.vec.remaining";
  v8 = *(_QWORD *)(a1 + 480);
  v54 = 259;
  v9 = sub_929DE0(&v55, v5, *(_BYTE **)(v8 + 64), (__int64)v53, 0, 0);
  v10 = *(_QWORD *)(a1 + 384);
  v11 = v9;
  v12 = *(_QWORD *)(a1 + 480);
  v13 = *(_BYTE *)(v12 + 16) == 0;
  v14 = *(_DWORD *)(v12 + 12);
  if ( v13 || (v15 = 1, !v14) )
    v15 = v14 > 1;
  v16 = sub_2AB31C0(v10, v15);
  v52 = 1;
  v51 = 3;
  v17 = *(_QWORD *)(v11 + 8);
  v50 = "min.epilog.iters.check";
  v18 = 36 - ((v16 == 0) - 1);
  v47 = sub_2AB26E0(
          (__int64)&v55,
          v17,
          *(_QWORD *)(*(_QWORD *)(a1 + 480) + 12LL),
          *(_DWORD *)(*(_QWORD *)(a1 + 480) + 20LL));
  v19 = (_QWORD *)(*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v62 + 7))(v62, v18, v11, v47);
  if ( !v19 )
  {
    v54 = 257;
    v19 = sub_BD2C40(72, unk_3F10FD0);
    if ( v19 )
    {
      v38 = *(_QWORD ***)(v11 + 8);
      v39 = *((unsigned __int8 *)v38 + 8);
      if ( (unsigned int)(v39 - 17) > 1 )
      {
        v41 = sub_BCB2A0(*v38);
      }
      else
      {
        BYTE4(v49) = (_BYTE)v39 == 18;
        LODWORD(v49) = *((_DWORD *)v38 + 8);
        v40 = (__int64 *)sub_BCB2A0(*v38);
        v41 = sub_BCE1B0(v40, v49);
      }
      sub_B523C0((__int64)v19, v41, 53, v18, v11, v47, (__int64)v53, 0, 0, 0);
    }
    (*((void (__fastcall **)(void **, _QWORD *, const char **, __int64, __int64))*v63 + 2))(v63, v19, &v50, v59, v60);
    v42 = v55;
    v43 = &v55[4 * (unsigned int)v56];
    if ( v55 != v43 )
    {
      do
      {
        v44 = *((_QWORD *)v42 + 1);
        v45 = *v42;
        v42 += 4;
        sub_B99FD0((__int64)v19, v45, v44);
      }
      while ( v43 != v42 );
    }
  }
  v46 = *(_QWORD *)(a1 + 240);
  v20 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v22 = v20;
  if ( v20 )
    sub_B4C9A0((__int64)v20, a2, v46, (__int64)v19, 3u, v21, 0, 0);
  v23 = sub_D47930(*(_QWORD *)(a1 + 8));
  v24 = sub_986580(v23);
  if ( (unsigned __int8)sub_BC8700(v24) )
  {
    v25 = *(_DWORD *)(a1 + 72) * *(_DWORD *)(a1 + 88);
    v26 = *(_DWORD *)(*(_QWORD *)(a1 + 480) + 12LL) * *(_DWORD *)(*(_QWORD *)(a1 + 480) + 20LL);
    if ( v26 > v25 )
      v26 = *(_DWORD *)(a1 + 72) * *(_DWORD *)(a1 + 88);
    v53[0] = v26;
    v53[1] = v25 - v26;
    sub_BC8EC0((__int64)v22, v53, 2, 0);
  }
  v27 = sub_986580(a3);
  sub_F34910(v27, v22);
  v30 = *(unsigned int *)(a1 + 272);
  if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 276) )
  {
    sub_C8D5F0(a1 + 264, (const void *)(a1 + 280), v30 + 1, 8u, v28, v29);
    v30 = *(unsigned int *)(a1 + 272);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8 * v30) = a3;
  v31 = *(_QWORD *)(a1 + 464);
  ++*(_DWORD *)(a1 + 272);
  v32 = sub_2BF0CC0(v31, a3);
  sub_2AB1780(**(_QWORD **)(a1 + 464), v32);
  **(_QWORD **)(a1 + 464) = v32;
  sub_2BF04F0(v32);
  sub_2AB95C0(a1, a3, v33, v34, v35, v36);
  nullsub_61();
  v70 = &unk_49DA100;
  nullsub_63();
  if ( v55 != (unsigned int *)v57 )
    _libc_free((unsigned __int64)v55);
  return a3;
}
