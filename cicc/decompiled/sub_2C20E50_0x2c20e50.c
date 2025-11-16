// Function: sub_2C20E50
// Address: 0x2c20e50
//
_QWORD *__fastcall sub_2C20E50(__int64 a1, __int64 a2)
{
  char v3; // cl
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int16 v11; // dx
  __int64 v12; // rsi
  unsigned __int8 v13; // cl
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned __int8 *v23; // rax
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r10
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v31; // rax
  __int16 v32; // ax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // r10
  __int64 v38; // rsi
  __int64 v39; // [rsp+0h] [rbp-110h]
  __int64 v40; // [rsp+8h] [rbp-108h]
  char v41; // [rsp+10h] [rbp-100h]
  __int64 v42; // [rsp+10h] [rbp-100h]
  _QWORD *v43; // [rsp+20h] [rbp-F0h]
  __int64 v44; // [rsp+20h] [rbp-F0h]
  __int64 v45; // [rsp+28h] [rbp-E8h]
  __int64 v46; // [rsp+28h] [rbp-E8h]
  __int64 v47; // [rsp+28h] [rbp-E8h]
  __int64 v48; // [rsp+28h] [rbp-E8h]
  _QWORD *v49; // [rsp+28h] [rbp-E8h]
  __int64 v50; // [rsp+28h] [rbp-E8h]
  __int64 v51; // [rsp+38h] [rbp-D8h]
  _BYTE v52[32]; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v53; // [rsp+60h] [rbp-B0h]
  __int64 v54[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v55; // [rsp+90h] [rbp-80h]
  const char *v56; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD v57[3]; // [rsp+A8h] [rbp-68h] BYREF
  __int64 v58; // [rsp+C0h] [rbp-50h]
  __int16 v59; // [rsp+C8h] [rbp-48h]
  _QWORD v60[8]; // [rsp+D0h] [rbp-40h] BYREF

  v3 = *(_BYTE *)(a2 + 12);
  v4 = *(_QWORD *)(a2 + 904);
  v5 = *(_DWORD *)(a2 + 8);
  BYTE4(v51) = v3;
  v40 = v4;
  LODWORD(v51) = v5 / *(_DWORD *)(a1 + 164);
  if ( !*(_DWORD *)(a1 + 56) )
    BUG();
  v7 = *(_QWORD *)(**(_QWORD **)(a1 + 48) + 40LL);
  v39 = **(_QWORD **)(a1 + 48);
  v41 = (v5 == 1) & (v3 ^ 1);
  v45 = *(_QWORD *)(v7 + 8);
  if ( !v41 )
  {
    v41 = *(_BYTE *)(a1 + 160);
    if ( !v41 )
      v45 = sub_BCE1B0((__int64 *)v45, v51);
  }
  v8 = *(_QWORD *)(a2 + 104);
  v56 = "vec.phi";
  LOWORD(v58) = 259;
  v9 = sub_BD2DA0(80);
  v10 = v9;
  if ( v9 )
  {
    v43 = (_QWORD *)v9;
    sub_B44260(v9, v45, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v10 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v10, &v56);
    sub_BD2A10(v10, *(_DWORD *)(v10 + 72), 1);
  }
  else
  {
    v43 = 0;
  }
  v12 = sub_AA5190(v8);
  if ( v12 )
  {
    v13 = v11;
    v14 = HIBYTE(v11);
  }
  else
  {
    v14 = 0;
    v13 = 0;
  }
  v15 = v13;
  BYTE1(v15) = v14;
  sub_B44220(v43, v12, v15);
  v16 = v10;
  LODWORD(v10) = 0;
  sub_2BF26E0(a2, a1 + 96, v16, *(_BYTE *)(a1 + 160));
  v44 = sub_2BF3650(a2 + 96, a1);
  v18 = *(_QWORD *)(a1 + 152);
  v19 = *(unsigned int *)(v18 + 40);
  if ( *(_DWORD *)(a1 + 56) == 3 )
  {
    v20 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL);
    if ( v20 )
    {
      v21 = *(_QWORD *)(v20 + 40);
      v10 = *(_QWORD *)(v21 + 24);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
        v10 = *(_QWORD *)v10;
    }
  }
  if ( (unsigned int)v19 <= 0x12 )
  {
    v29 = 455616;
    if ( _bittest64(&v29, v19) )
    {
      v24 = v7;
      if ( !v41 )
      {
        v56 = (const char *)v40;
        v31 = *(_QWORD *)(v40 + 48);
        v57[0] = 0;
        v57[1] = 0;
        v57[2] = v31;
        if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
          sub_BD73F0((__int64)v57);
        v32 = *(_WORD *)(v40 + 64);
        v58 = *(_QWORD *)(v40 + 56);
        v59 = v32;
        sub_B33910(v60, (__int64 *)v40);
        v33 = sub_986580(v44);
        sub_D5F1F0(v40, v33);
        v7 = sub_2BFB640(a2, v39, 0);
        sub_F11320((__int64)&v56);
        v24 = v7;
      }
      goto LABEL_23;
    }
  }
  else if ( (unsigned int)(v19 - 19) <= 1 )
  {
    v24 = v7;
    if ( !v41 )
    {
      sub_11A12D0((__int64)&v56, v40);
      v34 = sub_986580(v44);
      sub_D5F1F0(v40, v34);
      v35 = *(_QWORD *)(a2 + 8);
      v55 = 257;
      v7 = sub_B37620((unsigned int **)v40, v35, v7, v54);
      sub_F11320((__int64)&v56);
      v24 = v7;
    }
    goto LABEL_23;
  }
  v22 = v45;
  if ( (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17 <= 1 )
    v45 = **(_QWORD **)(v45 + 16);
  v23 = sub_F70230(v19, v45, *(unsigned int *)(v18 + 44), v22, v17);
  v24 = (__int64)v23;
  if ( !v41 )
  {
    LOWORD(v58) = 257;
    if ( (_DWORD)v10 )
    {
      v7 = sub_B37620((unsigned int **)v40, v51, (__int64)v23, (__int64 *)&v56);
      v28 = sub_2BFB640(a2, a1 + 96, *(_BYTE *)(a1 + 160));
    }
    else
    {
      v46 = sub_B37620((unsigned int **)v40, v51, (__int64)v23, (__int64 *)&v56);
      sub_11A12D0((__int64)&v56, v40);
      v25 = sub_986580(v44);
      sub_D5F1F0(v40, v25);
      v26 = sub_BCB2D0(*(_QWORD **)(v40 + 72));
      v53 = 257;
      v42 = sub_ACD640(v26, 0, 0);
      v27 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v40 + 80) + 104LL))(
              *(_QWORD *)(v40 + 80),
              v46,
              v7);
      if ( !v27 )
      {
        v55 = 257;
        v36 = sub_BD2C40(72, 3u);
        v37 = v36;
        if ( v36 )
        {
          v38 = v46;
          v49 = v36;
          sub_B4DFA0((__int64)v36, v38, v7, v42, (__int64)v54, 0, 0, 0);
          v37 = v49;
        }
        v50 = (__int64)v37;
        (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v40 + 88) + 16LL))(
          *(_QWORD *)(v40 + 88),
          v37,
          v52,
          *(_QWORD *)(v40 + 56),
          *(_QWORD *)(v40 + 64));
        sub_94AAF0((unsigned int **)v40, v50);
        v27 = v50;
      }
      v47 = v27;
      sub_F11320((__int64)&v56);
      v28 = sub_2BFB640(a2, a1 + 96, *(_BYTE *)(a1 + 160));
      v7 = v47;
    }
    return sub_F0A850(v28, v7, v44);
  }
LABEL_23:
  v48 = v24;
  v28 = sub_2BFB640(a2, a1 + 96, *(_BYTE *)(a1 + 160));
  if ( (_DWORD)v10 )
    v7 = v48;
  return sub_F0A850(v28, v7, v44);
}
