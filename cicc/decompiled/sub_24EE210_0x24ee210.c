// Function: sub_24EE210
// Address: 0x24ee210
//
__int64 __fastcall sub_24EE210(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rsi
  _QWORD *v11; // r12
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rsi
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // rsi
  __int64 v21; // [rsp+10h] [rbp-1D0h]
  __int64 v22; // [rsp+18h] [rbp-1C8h]
  _QWORD v23[2]; // [rsp+28h] [rbp-1B8h] BYREF
  __int64 v24; // [rsp+38h] [rbp-1A8h]
  __int64 v25; // [rsp+40h] [rbp-1A0h]
  void *v26; // [rsp+50h] [rbp-190h]
  _QWORD v27[2]; // [rsp+58h] [rbp-188h] BYREF
  __int64 v28; // [rsp+68h] [rbp-178h]
  __int64 v29; // [rsp+70h] [rbp-170h]
  _QWORD v30[4]; // [rsp+80h] [rbp-160h] BYREF
  int v31; // [rsp+A0h] [rbp-140h]
  _BYTE *v32; // [rsp+A8h] [rbp-138h]
  __int64 v33; // [rsp+B0h] [rbp-130h]
  _BYTE v34[32]; // [rsp+B8h] [rbp-128h] BYREF
  __int64 v35; // [rsp+D8h] [rbp-108h]
  __int64 v36; // [rsp+E0h] [rbp-100h]
  __int16 v37; // [rsp+E8h] [rbp-F8h]
  __int64 v38; // [rsp+F0h] [rbp-F0h]
  void **v39; // [rsp+F8h] [rbp-E8h]
  _QWORD *v40; // [rsp+100h] [rbp-E0h]
  __int64 v41; // [rsp+108h] [rbp-D8h]
  int v42; // [rsp+110h] [rbp-D0h]
  __int16 v43; // [rsp+114h] [rbp-CCh]
  char v44; // [rsp+116h] [rbp-CAh]
  __int64 v45; // [rsp+118h] [rbp-C8h]
  __int64 v46; // [rsp+120h] [rbp-C0h]
  void *v47; // [rsp+128h] [rbp-B8h] BYREF
  _QWORD v48[3]; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+148h] [rbp-98h] BYREF
  _QWORD *v50; // [rsp+150h] [rbp-90h]
  unsigned int v51; // [rsp+160h] [rbp-80h]
  _QWORD *v52; // [rsp+170h] [rbp-70h]
  unsigned int v53; // [rsp+180h] [rbp-60h]
  char v54; // [rsp+188h] [rbp-58h]
  __int64 v55; // [rsp+198h] [rbp-48h]
  __int64 v56; // [rsp+1A0h] [rbp-40h]
  __int64 v57; // [rsp+1A8h] [rbp-38h]

  v30[1] = a1;
  v22 = sub_C996C0("SwitchCloner", 12, 0, 0);
  v30[2] = a2;
  v30[3] = a3;
  v31 = a4;
  v38 = sub_B2BE50(a1);
  v43 = 512;
  v32 = v34;
  v47 = &unk_49DA100;
  v33 = 0x200000000LL;
  v37 = 0;
  v48[1] = a5;
  v39 = &v47;
  v40 = v48;
  v41 = 0;
  v42 = 0;
  v44 = 7;
  v45 = 0;
  v46 = 0;
  v35 = 0;
  v36 = 0;
  v48[0] = &unk_49DA0B0;
  v48[2] = a6;
  v49 = 0;
  v51 = 128;
  v50 = (_QWORD *)sub_C7D670(0x2000, 8);
  sub_23FE7B0((__int64)&v49);
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v30[0] = &unk_4A16A88;
  v57 = 0;
  sub_24EE180((__int64)v30);
  v30[0] = &unk_4A16A60;
  v21 = v55;
  if ( v54 )
  {
    v16 = v53;
    v54 = 0;
    if ( v53 )
    {
      v17 = v52;
      v18 = &v52[2 * v53];
      do
      {
        if ( *v17 != -4096 && *v17 != -8192 )
        {
          v19 = v17[1];
          if ( v19 )
            sub_B91220((__int64)(v17 + 1), v19);
        }
        v17 += 2;
      }
      while ( v18 != v17 );
      v16 = v53;
    }
    sub_C7D6A0((__int64)v52, 16 * v16, 8);
  }
  v9 = v51;
  if ( v51 )
  {
    v11 = v50;
    v23[0] = 2;
    v23[1] = 0;
    v12 = &v50[8 * (unsigned __int64)v51];
    v24 = -4096;
    v26 = &unk_49DD7B0;
    v13 = -4096;
    v25 = 0;
    v27[0] = 2;
    v27[1] = 0;
    v28 = -8192;
    v29 = 0;
    while ( 1 )
    {
      v14 = v11[3];
      if ( v14 != v13 )
      {
        v13 = v28;
        if ( v14 != v28 )
        {
          v15 = v11[7];
          if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
          {
            sub_BD60C0(v11 + 5);
            v14 = v11[3];
          }
          v13 = v14;
        }
      }
      *v11 = &unk_49DB368;
      if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
        sub_BD60C0(v11 + 1);
      v11 += 8;
      if ( v12 == v11 )
        break;
      v13 = v24;
    }
    v26 = &unk_49DB368;
    if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
      sub_BD60C0(v27);
    if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
      sub_BD60C0(v23);
    v9 = v51;
  }
  sub_C7D6A0((__int64)v50, v9 << 6, 8);
  nullsub_61();
  v47 = &unk_49DA100;
  nullsub_63();
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  if ( v22 )
    sub_C9AF60(v22);
  return v21;
}
