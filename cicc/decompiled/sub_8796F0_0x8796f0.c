// Function: sub_8796F0
// Address: 0x8796f0
//
__int64 __fastcall sub_8796F0(__int64 a1)
{
  __int64 v2; // r13
  __int64 *v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 *v8; // r9
  bool v9; // zf
  __int64 v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  const __m128i *v15; // rdi
  __int64 v16; // r8
  __int64 *v17; // r9
  __int64 v18; // rcx
  __int64 *v19; // rdx
  __int64 v20; // r15
  __int64 v21; // r10
  __int64 v22; // r14
  __int64 v23; // rsi
  unsigned int v24; // r12d
  __int64 v25; // rsi
  __int64 i; // rbx
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // r10
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 *v33; // [rsp-10h] [rbp-2D0h]
  __int64 v34; // [rsp-8h] [rbp-2C8h]
  __int64 v35; // [rsp+0h] [rbp-2C0h]
  __int64 v36; // [rsp+8h] [rbp-2B8h]
  unsigned int v37; // [rsp+1Ch] [rbp-2A4h] BYREF
  const __m128i *v38; // [rsp+20h] [rbp-2A0h] BYREF
  const __m128i *v39; // [rsp+28h] [rbp-298h] BYREF
  __int64 v40; // [rsp+30h] [rbp-290h] BYREF
  __int64 v41; // [rsp+38h] [rbp-288h]
  __int64 v42; // [rsp+40h] [rbp-280h]
  __m128i v43[6]; // [rsp+50h] [rbp-270h] BYREF
  _QWORD v44[66]; // [rsp+B0h] [rbp-210h] BYREF

  v2 = *(_QWORD *)(a1 + 88);
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v3 = *(__int64 **)(v2 + 216);
  v4 = sub_823970(24);
  *(_BYTE *)(a1 + 104) &= ~1u;
  v9 = *(_BYTE *)(a1 + 80) == 10;
  v40 = v4;
  v41 = 1;
  v37 = 0;
  if ( v9 )
  {
    v10 = v4;
    v11 = *(__int64 **)(v2 + 248);
    if ( !v11 )
    {
      v37 = 1;
      v25 = 24;
      v24 = 1;
      goto LABEL_12;
    }
    v12 = *(_QWORD *)(a1 + 64);
    sub_865900(*v11);
    memset(v44, 0, 0x1D8u);
    v44[19] = v44;
    v44[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v44[22]) |= 1u;
    v13 = *(_QWORD *)(v2 + 152);
    BYTE1(v44[16]) |= 0x40u;
    v44[0] = a1;
    v44[36] = v13;
    sub_8600D0(1u, -1, *(_QWORD *)(v2 + 152), 0);
    v14 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_BYTE *)(v14 + 11) |= 0x40u;
    *(_QWORD *)(v14 + 624) = v44;
  }
  else
  {
    v12 = sub_5EA620(*(_QWORD *)(v2 + 344));
    for ( i = v12; !*(_QWORD *)(*(_QWORD *)(i + 168) + 160LL); i = *(_QWORD *)(*(_QWORD *)(i + 40) + 32LL) )
      ;
    sub_8646E0(v12, 0);
    sub_865900(**(_QWORD **)(*(_QWORD *)(i + 168) + 160LL));
  }
  v15 = (const __m128i *)v12;
  sub_89F970(v12, &v40);
  v9 = v42 == 1;
  v18 = 3 * (v42 - 1);
  v19 = (__int64 *)(v40 + 24 * (v42 - 1));
  v20 = *v19;
  v21 = v19[1];
  --v42;
  if ( v9 )
  {
    v22 = *v3;
  }
  else
  {
    v35 = v21;
    v39 = 0;
    v38 = (const __m128i *)sub_724DC0();
    sub_892150(v43);
    v29 = sub_6EFFF0(*v3, &v40, (__int64)v43, 0x4000, v38, (__int64 *)&v39, &v37);
    v22 = v29;
    if ( v37 || v29 )
    {
      v15 = (const __m128i *)&v38;
      sub_724E30((__int64)&v38);
      v21 = v35;
    }
    else
    {
      if ( v39 )
      {
        sub_724E30((__int64)&v38);
        v15 = v39;
        v30 = v35;
      }
      else
      {
        v32 = sub_724E50((__int64 *)&v38, &v40);
        v30 = v35;
        v39 = (const __m128i *)v32;
        v15 = (const __m128i *)v32;
      }
      v36 = v30;
      v31 = sub_73A720(v15, (__int64)&v40);
      v21 = v36;
      v22 = (__int64)v31;
    }
  }
  v23 = v37;
  if ( !v37 )
  {
    v23 = v21;
    v15 = (const __m128i *)v22;
    v43[0] = 0u;
    v28 = sub_6F1C10(v22, v21, v20, v43, 0x40000u, 0, (int *)&v37, 0);
    v19 = v33;
    v18 = v34;
    if ( v28 )
    {
      if ( !v37 )
      {
        if ( *(_BYTE *)(a1 + 80) != 10 )
          goto LABEL_9;
LABEL_20:
        sub_863FC0((__int64)v15, v23, (__int64)v19, v18, v16, v17);
        goto LABEL_9;
      }
    }
    else
    {
      v37 = 1;
    }
  }
  *(_BYTE *)(v2 + 208) |= 4u;
  if ( *(_BYTE *)(a1 + 80) == 10 )
    goto LABEL_20;
LABEL_9:
  sub_864110((__int64)v15, v23, (__int64)v19, v18, v16, v17);
  if ( *(_BYTE *)(a1 + 80) != 10 )
    sub_866010();
  v24 = v37;
  v10 = v40;
  v25 = 24 * v41;
LABEL_12:
  sub_823A00(v10, v25, v5, v6, v7, v8);
  return v24;
}
