// Function: sub_2018D60
// Address: 0x2018d60
//
__int64 __fastcall sub_2018D60(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, __m128i a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  char v10; // dl
  const void **v11; // rax
  int v12; // r13d
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  char v17; // dl
  const void **v18; // rax
  int v19; // r8d
  _BYTE *v20; // rax
  __int64 v21; // r9
  _BYTE *v22; // r10
  _BYTE *v23; // rdx
  _BYTE *v24; // rax
  int v25; // ecx
  int v26; // edx
  __int64 v27; // rcx
  const void *v28; // r12
  __int64 v29; // r13
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r10
  __int128 v33; // rax
  __int64 v34; // r12
  int v36; // [rsp+0h] [rbp-E0h]
  _QWORD *v37; // [rsp+0h] [rbp-E0h]
  int v38; // [rsp+0h] [rbp-E0h]
  __int64 v39; // [rsp+8h] [rbp-D8h]
  int v40; // [rsp+10h] [rbp-D0h]
  int v41; // [rsp+10h] [rbp-D0h]
  _QWORD *v42; // [rsp+10h] [rbp-D0h]
  __int64 *v43; // [rsp+10h] [rbp-D0h]
  __int64 v44; // [rsp+10h] [rbp-D0h]
  __int64 v45; // [rsp+20h] [rbp-C0h] BYREF
  int v46; // [rsp+28h] [rbp-B8h]
  unsigned int v47; // [rsp+30h] [rbp-B0h] BYREF
  const void **v48; // [rsp+38h] [rbp-A8h]
  __int64 v49; // [rsp+40h] [rbp-A0h] BYREF
  const void **v50; // [rsp+48h] [rbp-98h]
  __int64 v51; // [rsp+50h] [rbp-90h] BYREF
  int v52; // [rsp+58h] [rbp-88h]
  _BYTE *v53; // [rsp+60h] [rbp-80h] BYREF
  __int64 v54; // [rsp+68h] [rbp-78h]
  _BYTE s[112]; // [rsp+70h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v45 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v45, v8, 2);
  v46 = *(_DWORD *)(a2 + 64);
  v9 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v10 = *(_BYTE *)v9;
  v11 = *(const void ***)(v9 + 8);
  LOBYTE(v47) = v10;
  v48 = v11;
  if ( v10 )
    v12 = word_4301260[(unsigned __int8)(v10 - 14)];
  else
    v12 = sub_1F58D30((__int64)&v47);
  v13 = *(__int64 **)(a2 + 32);
  v14 = *v13;
  v15 = v13[1];
  v16 = *(_QWORD *)(*v13 + 40) + 16LL * *((unsigned int *)v13 + 2);
  v17 = *(_BYTE *)v16;
  v18 = *(const void ***)(v16 + 8);
  LOBYTE(v49) = v17;
  v50 = v18;
  if ( v17 )
    v19 = word_4301260[(unsigned __int8)(v17 - 14)];
  else
    v19 = sub_1F58D30((__int64)&v49);
  v20 = s;
  v21 = v19;
  v53 = s;
  v54 = 0x1000000000LL;
  if ( v19 )
  {
    v22 = s;
    if ( (unsigned __int64)v19 > 0x10 )
    {
      v38 = v19;
      v44 = v19;
      sub_16CD150((__int64)&v53, s, v19, 4, v19, v19);
      v20 = v53;
      v19 = v38;
      v21 = v44;
      v22 = &v53[4 * (unsigned int)v54];
    }
    v23 = &v20[4 * v21];
    if ( v23 != v22 )
    {
      v36 = v19;
      v40 = v21;
      memset(v22, 255, v23 - v22);
      LODWORD(v21) = v40;
      v19 = v36;
    }
    LODWORD(v54) = v21;
  }
  v41 = v19 / v12;
  v24 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
  v25 = v41 - 1;
  if ( !*v24 )
    v25 = 0;
  if ( v12 > 0 )
  {
    v26 = 0;
    v27 = 4LL * v25;
    do
    {
      *(_DWORD *)&v53[v27] = v26++;
      v27 += 4LL * v41;
    }
    while ( v12 != v26 );
  }
  v28 = v53;
  v29 = (unsigned int)v54;
  v42 = *(_QWORD **)a1;
  v51 = 0;
  v52 = 0;
  v30 = sub_1D2B300(v42, 0x30u, (__int64)&v51, v49, (__int64)v50, (__int64)&v51);
  v32 = (__int64)v42;
  if ( v51 )
  {
    v37 = v30;
    v39 = v31;
    sub_161E7C0((__int64)&v51, v51);
    v30 = v37;
    v31 = v39;
    v32 = (__int64)v42;
  }
  v43 = (__int64 *)v32;
  *(_QWORD *)&v33 = sub_1D41320(
                      v32,
                      (unsigned int)v49,
                      v50,
                      (__int64)&v45,
                      v14,
                      v15,
                      a4,
                      a5,
                      a6,
                      (__int64)v30,
                      v31,
                      v28,
                      v29);
  v34 = sub_1D309E0(v43, 158, (__int64)&v45, v47, v48, 0, a4, a5, *(double *)a6.m128i_i64, v33);
  if ( v53 != s )
    _libc_free((unsigned __int64)v53);
  if ( v45 )
    sub_161E7C0((__int64)&v45, v45);
  return v34;
}
