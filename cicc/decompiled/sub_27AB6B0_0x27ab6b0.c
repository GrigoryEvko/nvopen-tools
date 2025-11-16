// Function: sub_27AB6B0
// Address: 0x27ab6b0
//
__int64 __fastcall sub_27AB6B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  void *v9; // rsi
  unsigned __int64 v10; // r15
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r13
  __int64 v16; // rax
  __int64 v18; // [rsp+8h] [rbp-328h]
  __int64 v19; // [rsp+10h] [rbp-320h]
  __int64 v20; // [rsp+18h] [rbp-318h]
  __int64 v21; // [rsp+20h] [rbp-310h] BYREF
  _QWORD *v22; // [rsp+28h] [rbp-308h]
  int v23; // [rsp+30h] [rbp-300h]
  int v24; // [rsp+34h] [rbp-2FCh]
  int v25; // [rsp+38h] [rbp-2F8h]
  char v26; // [rsp+3Ch] [rbp-2F4h]
  _QWORD v27[2]; // [rsp+40h] [rbp-2F0h] BYREF
  __int64 v28; // [rsp+50h] [rbp-2E0h] BYREF
  _BYTE *v29; // [rsp+58h] [rbp-2D8h]
  __int64 v30; // [rsp+60h] [rbp-2D0h]
  int v31; // [rsp+68h] [rbp-2C8h]
  char v32; // [rsp+6Ch] [rbp-2C4h]
  _BYTE v33[16]; // [rsp+70h] [rbp-2C0h] BYREF
  _BYTE v34[216]; // [rsp+80h] [rbp-2B0h] BYREF
  __int64 v35; // [rsp+158h] [rbp-1D8h]
  __int64 v36; // [rsp+160h] [rbp-1D0h]
  __int64 v37; // [rsp+168h] [rbp-1C8h]
  __int64 v38; // [rsp+170h] [rbp-1C0h]
  __int64 v39; // [rsp+178h] [rbp-1B8h]
  unsigned __int64 v40; // [rsp+180h] [rbp-1B0h]
  __int64 v41; // [rsp+188h] [rbp-1A8h]
  __int64 v42; // [rsp+190h] [rbp-1A0h]
  __int64 v43; // [rsp+198h] [rbp-198h]
  unsigned int v44; // [rsp+1A0h] [rbp-190h]
  __int64 v45; // [rsp+1A8h] [rbp-188h]
  __int64 v46; // [rsp+1B0h] [rbp-180h]
  __int64 v47; // [rsp+1B8h] [rbp-178h]
  unsigned int v48; // [rsp+1C0h] [rbp-170h]
  __int64 v49; // [rsp+1C8h] [rbp-168h]
  __int64 v50; // [rsp+1D0h] [rbp-160h]
  __int64 v51; // [rsp+1D8h] [rbp-158h]
  __int64 v52; // [rsp+1E0h] [rbp-150h]
  _BYTE *v53; // [rsp+1E8h] [rbp-148h]
  __int64 v54; // [rsp+1F0h] [rbp-140h]
  _BYTE v55[312]; // [rsp+1F8h] [rbp-138h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v18 = sub_BC1CD0(a4, &unk_4F8FBC8, a3);
  v19 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v20 = sub_BC1CD0(a4, &unk_4F8EE60, a3);
  v7 = *(_QWORD *)(sub_BC1CD0(a4, &unk_4F8F810, a3) + 8);
  sub_278A360((__int64)v34);
  v35 = v6 + 8;
  v39 = v7;
  v37 = v19 + 8;
  v36 = v18 + 8;
  v38 = v20 + 8;
  v8 = sub_22077B0(0x2F8u);
  if ( v8 )
  {
    *(_QWORD *)v8 = v7;
    *(_QWORD *)(v8 + 8) = v8 + 24;
    *(_QWORD *)(v8 + 416) = v8 + 440;
    *(_QWORD *)(v8 + 504) = v8 + 520;
    *(_QWORD *)(v8 + 16) = 0x1000000000LL;
    *(_QWORD *)(v8 + 408) = 0;
    *(_QWORD *)(v8 + 424) = 8;
    *(_DWORD *)(v8 + 432) = 0;
    *(_BYTE *)(v8 + 436) = 1;
    *(_QWORD *)(v8 + 512) = 0x800000000LL;
    *(_DWORD *)(v8 + 720) = 0;
    *(_QWORD *)(v8 + 728) = 0;
    *(_QWORD *)(v8 + 736) = v8 + 720;
    *(_QWORD *)(v8 + 744) = v8 + 720;
    *(_QWORD *)(v8 + 752) = 0;
  }
  v55[260] = 0;
  v40 = v8;
  v54 = 0x2000000000LL;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = v55;
  sub_1047D20(v7);
  v9 = (void *)(a1 + 32);
  if ( !(unsigned __int8)sub_27AAC70((__int64)v34, a3) )
  {
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_5;
  }
  v23 = 2;
  v22 = v27;
  v27[0] = &unk_4F81450;
  v25 = 0;
  v26 = 1;
  v28 = 0;
  v29 = v33;
  v30 = 2;
  v31 = 0;
  v32 = 1;
  v24 = 1;
  v21 = 1;
  if ( &unk_4F81450 != (_UNKNOWN *)&qword_4F82400 && &unk_4F81450 != &unk_4F8F810 )
  {
    v24 = 2;
    v27[1] = &unk_4F8F810;
    v21 = 2;
  }
  sub_C8CF70(a1, v9, 2, (__int64)v27, (__int64)&v21);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v33, (__int64)&v28);
  if ( v32 )
  {
    if ( v26 )
      goto LABEL_5;
    goto LABEL_33;
  }
  _libc_free((unsigned __int64)v29);
  if ( !v26 )
LABEL_33:
    _libc_free((unsigned __int64)v22);
LABEL_5:
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  sub_C7D6A0(v50, 8LL * (unsigned int)v52, 8);
  sub_C7D6A0(v46, 16LL * v48, 8);
  sub_C7D6A0(v42, 16LL * v44, 8);
  v10 = v40;
  if ( v40 )
  {
    sub_27A10F0(*(_QWORD **)(v40 + 728));
    v11 = *(_QWORD **)(v10 + 504);
    v12 = &v11[3 * *(unsigned int *)(v10 + 512)];
    if ( v11 != v12 )
    {
      do
      {
        v13 = *(v12 - 1);
        v12 -= 3;
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD60C0(v12);
      }
      while ( v11 != v12 );
      v12 = *(_QWORD **)(v10 + 504);
    }
    if ( v12 != (_QWORD *)(v10 + 520) )
      _libc_free((unsigned __int64)v12);
    if ( !*(_BYTE *)(v10 + 436) )
      _libc_free(*(_QWORD *)(v10 + 416));
    v14 = *(_QWORD **)(v10 + 8);
    v15 = &v14[3 * *(unsigned int *)(v10 + 16)];
    if ( v14 != v15 )
    {
      do
      {
        v16 = *(v15 - 1);
        v15 -= 3;
        if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
          sub_BD60C0(v15);
      }
      while ( v14 != v15 );
      v15 = *(_QWORD **)(v10 + 8);
    }
    if ( v15 != (_QWORD *)(v10 + 24) )
      _libc_free((unsigned __int64)v15);
    j_j___libc_free_0(v10);
  }
  sub_278E4C0((__int64)v34);
  return a1;
}
