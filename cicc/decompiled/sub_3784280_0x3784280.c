// Function: sub_3784280
// Address: 0x3784280
//
void __fastcall sub_3784280(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  unsigned int v9; // eax
  unsigned __int64 v10; // r15
  unsigned int *i; // rax
  unsigned int *v12; // rdx
  unsigned __int16 **v13; // r8
  __int64 v14; // rbx
  unsigned int v15; // r14d
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  unsigned int *v19; // rax
  __int64 v20; // rax
  unsigned int *v21; // rax
  __int64 v22; // r14
  unsigned __int16 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r10
  _WORD *v26; // rax
  __int64 v27; // rsi
  _QWORD *v28; // rdi
  unsigned __int8 *v29; // rax
  __int64 v30; // r9
  unsigned __int8 *v31; // rax
  __int64 v32; // r9
  unsigned __int64 v33; // r8
  unsigned int v34; // ebx
  __int64 v35; // r15
  unsigned __int64 v36; // r13
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  unsigned __int16 *v39; // rax
  unsigned __int16 *v40; // rdx
  __int128 v41; // [rsp-20h] [rbp-1D0h]
  __int128 v42; // [rsp-10h] [rbp-1C0h]
  unsigned __int64 v43; // [rsp+8h] [rbp-1A8h]
  unsigned __int64 v44; // [rsp+28h] [rbp-188h]
  unsigned __int64 v45; // [rsp+30h] [rbp-180h]
  unsigned __int64 v46; // [rsp+38h] [rbp-178h]
  __int64 v47; // [rsp+38h] [rbp-178h]
  unsigned int v48; // [rsp+44h] [rbp-16Ch]
  unsigned __int16 **v49; // [rsp+48h] [rbp-168h]
  unsigned __int64 v50; // [rsp+48h] [rbp-168h]
  unsigned int v51; // [rsp+48h] [rbp-168h]
  __int64 v52; // [rsp+50h] [rbp-160h] BYREF
  unsigned int v53; // [rsp+58h] [rbp-158h]
  unsigned int *v54; // [rsp+60h] [rbp-150h] BYREF
  __int64 v55; // [rsp+68h] [rbp-148h]
  _OWORD v56[8]; // [rsp+70h] [rbp-140h] BYREF
  unsigned __int16 *v57; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+F8h] [rbp-B8h]
  _BYTE v59[176]; // [rsp+100h] [rbp-B0h] BYREF

  v9 = *(_DWORD *)(a2 + 64);
  v55 = 0x800000000LL;
  v48 = v9;
  v10 = 2 * v9;
  i = (unsigned int *)v56;
  v54 = (unsigned int *)v56;
  if ( v10 )
  {
    if ( v10 > 8 )
    {
      sub_C8D5F0((__int64)&v54, v56, v10, 0x10u, a5, a6);
      v12 = &v54[4 * v10];
      for ( i = &v54[4 * (unsigned int)v55]; v12 != i; i += 4 )
      {
LABEL_4:
        if ( i )
        {
          *(_QWORD *)i = 0;
          i[2] = 0;
        }
      }
    }
    else
    {
      v12 = (unsigned int *)&v56[v10];
      if ( v12 != (unsigned int *)v56 )
        goto LABEL_4;
    }
    LODWORD(v55) = v10;
  }
  if ( v48 )
  {
    v44 = v6;
    v13 = &v57;
    v14 = 0;
    v15 = 1;
    do
    {
      v16 = *(_QWORD *)(a2 + 40);
      v52 = 0;
      v53 = 0;
      v57 = 0;
      LODWORD(v58) = 0;
      v17 = *(_QWORD *)(v16 + v14 + 8);
      v18 = *(_QWORD *)(v16 + v14);
      v49 = v13;
      v14 += 40;
      sub_375E8D0(a1, v18, v17, (__int64)&v52, (__int64)v13);
      v13 = v49;
      v19 = &v54[4 * v15 - 4];
      *(_QWORD *)v19 = v52;
      v19[2] = v53;
      v20 = v15;
      v15 += 2;
      v21 = &v54[4 * v20];
      *(_QWORD *)v21 = v57;
      v21[2] = v58;
    }
    while ( 40LL * v48 != v14 );
    v22 = v48;
    v6 = v44;
    v23 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v54 + 48LL) + 16LL * v54[2]);
    v24 = v48;
    a6 = *v23;
    v25 = *((_QWORD *)v23 + 1);
    v26 = v59;
    v58 = 0x800000000LL;
    v57 = (unsigned __int16 *)v59;
    if ( v48 > 8 )
    {
      v47 = v25;
      v51 = a6;
      sub_C8D5F0((__int64)v13, v59, v48, 0x10u, (__int64)v13, a6);
      v39 = v57;
      a6 = v51;
      v40 = &v57[8 * v48];
      do
      {
        if ( v39 )
        {
          *v39 = v51;
          *((_QWORD *)v39 + 1) = v47;
        }
        v39 += 8;
      }
      while ( v39 != v40 );
    }
    else
    {
      do
      {
        *v26 = a6;
        v26 += 8;
        *((_QWORD *)v26 - 1) = v25;
        --v24;
      }
      while ( v24 );
    }
  }
  else
  {
    v22 = 0;
    HIDWORD(v58) = 8;
    v57 = (unsigned __int16 *)v59;
  }
  v27 = *(_QWORD *)(a2 + 80);
  LODWORD(v58) = v48;
  v52 = v27;
  if ( v27 )
    sub_B96E90((__int64)&v52, v27, 1);
  v28 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v42 + 1) = v22;
  v53 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v42 = v54;
  v29 = sub_3411BE0(v28, 0xA2u, (__int64)&v52, v57, (unsigned int)v58, a6, v42);
  *((_QWORD *)&v41 + 1) = v22;
  *(_QWORD *)&v41 = &v54[4 * v22];
  v46 = (unsigned __int64)v29;
  v31 = sub_3411BE0(*(_QWORD **)(a1 + 8), 0xA2u, (__int64)&v52, v57, (unsigned int)v58, v30, v41);
  v32 = 0;
  v45 = (unsigned __int64)v31;
  if ( v48 )
  {
    v33 = v6;
    v34 = 0;
    v35 = a1;
    v36 = v43;
    do
    {
      v37 = v34;
      v38 = v34 | v36 & 0xFFFFFFFF00000000LL;
      v50 = v34++ | v33 & 0xFFFFFFFF00000000LL;
      v36 = v38;
      sub_3760810(v35, a2, v37, v46, v50, v32, v45, v38);
      v33 = v50;
    }
    while ( v48 != v34 );
  }
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
  if ( v57 != (unsigned __int16 *)v59 )
    _libc_free((unsigned __int64)v57);
  if ( v54 != (unsigned int *)v56 )
    _libc_free((unsigned __int64)v54);
}
