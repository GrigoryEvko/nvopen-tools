// Function: sub_28B4370
// Address: 0x28b4370
//
__int64 __fastcall sub_28B4370(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __int64 a10,
        __int64 a11)
{
  unsigned int v11; // eax
  unsigned int v12; // r12d
  unsigned int v13; // eax
  _QWORD *v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 *v17; // rdx
  unsigned __int64 v18; // r13
  _QWORD *v19; // rbx
  _QWORD *v20; // r15
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r14
  __int64 v24; // rax
  _QWORD *v26; // [rsp+0h] [rbp-3A0h]
  _QWORD *v27; // [rsp+0h] [rbp-3A0h]
  _QWORD v28[4]; // [rsp+10h] [rbp-390h] BYREF
  __int64 v29; // [rsp+30h] [rbp-370h]
  __int64 v30; // [rsp+38h] [rbp-368h]
  unsigned int v31; // [rsp+40h] [rbp-360h]
  __int64 v32; // [rsp+48h] [rbp-358h]
  _QWORD *v33; // [rsp+50h] [rbp-350h]
  __int64 v34; // [rsp+58h] [rbp-348h]
  unsigned int v35; // [rsp+60h] [rbp-340h]
  __int64 v36; // [rsp+70h] [rbp-330h] BYREF
  _BYTE *v37; // [rsp+78h] [rbp-328h]
  __int64 v38; // [rsp+80h] [rbp-320h]
  _BYTE v39[384]; // [rsp+88h] [rbp-318h] BYREF
  __int64 v40; // [rsp+208h] [rbp-198h]
  char *v41; // [rsp+210h] [rbp-190h]
  __int64 v42; // [rsp+218h] [rbp-188h]
  int v43; // [rsp+220h] [rbp-180h]
  char v44; // [rsp+224h] [rbp-17Ch]
  char v45; // [rsp+228h] [rbp-178h] BYREF
  _BYTE *v46; // [rsp+268h] [rbp-138h]
  __int64 v47; // [rsp+270h] [rbp-130h]
  _BYTE v48[200]; // [rsp+278h] [rbp-128h] BYREF
  int v49; // [rsp+340h] [rbp-60h] BYREF
  _QWORD *v50; // [rsp+348h] [rbp-58h]
  int *v51; // [rsp+350h] [rbp-50h]
  int *v52; // [rsp+358h] [rbp-48h]
  __int64 v53; // [rsp+360h] [rbp-40h]

  *a1 = a3;
  a1[1] = a4;
  a1[4] = a10;
  a1[2] = a5;
  a1[5] = a11;
  v36 = a11;
  v37 = v39;
  v38 = 0x1000000000LL;
  v41 = &v45;
  v47 = 0x800000000LL;
  a1[3] = a6;
  v40 = 0;
  v42 = 8;
  v43 = 0;
  v44 = 1;
  v46 = v48;
  v49 = 0;
  v50 = 0;
  v51 = &v49;
  v52 = &v49;
  v53 = 0;
  a1[6] = &v36;
  v28[1] = a6;
  v28[2] = 0;
  v28[3] = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  a1[7] = v28;
  v28[0] = &unk_49DDC10;
  v11 = 0;
  do
  {
    v12 = v11;
    v11 = sub_28B3C30(a1, a2, a7, a8, a9);
  }
  while ( (_BYTE)v11 );
  if ( byte_4F8F8E8[0] )
    nullsub_390();
  v28[0] = &unk_49DDC10;
  v13 = v35;
  if ( v35 )
  {
    v14 = v33;
    v15 = &v33[2 * v35];
    do
    {
      if ( *v14 != -8192 && *v14 != -4096 )
      {
        v16 = v14[1];
        if ( v16 )
        {
          if ( (v16 & 4) != 0 )
          {
            v17 = (unsigned __int64 *)(v16 & 0xFFFFFFFFFFFFFFF8LL);
            v18 = (unsigned __int64)v17;
            if ( v17 )
            {
              if ( (unsigned __int64 *)*v17 != v17 + 2 )
              {
                v26 = v15;
                _libc_free(*v17);
                v15 = v26;
              }
              v27 = v15;
              j_j___libc_free_0(v18);
              v15 = v27;
            }
          }
        }
      }
      v14 += 2;
    }
    while ( v15 != v14 );
    v13 = v35;
  }
  sub_C7D6A0((__int64)v33, 16LL * v13, 8);
  sub_C7D6A0(v29, 16LL * v31, 8);
  nullsub_184();
  sub_28A9C40(v50);
  v19 = v46;
  v20 = &v46[24 * (unsigned int)v47];
  if ( v46 != (_BYTE *)v20 )
  {
    do
    {
      v21 = *(v20 - 1);
      v20 -= 3;
      if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
        sub_BD60C0(v20);
    }
    while ( v19 != v20 );
    v20 = v46;
  }
  if ( v20 != (_QWORD *)v48 )
    _libc_free((unsigned __int64)v20);
  if ( !v44 )
    _libc_free((unsigned __int64)v41);
  v22 = v37;
  v23 = &v37[24 * (unsigned int)v38];
  if ( v37 != (_BYTE *)v23 )
  {
    do
    {
      v24 = *(v23 - 1);
      v23 -= 3;
      if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
        sub_BD60C0(v23);
    }
    while ( v22 != v23 );
    v23 = v37;
  }
  if ( v23 != (_QWORD *)v39 )
    _libc_free((unsigned __int64)v23);
  return v12;
}
