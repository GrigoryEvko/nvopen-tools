// Function: sub_3777490
// Address: 0x3777490
//
void __fastcall sub_3777490(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        __m128i a7,
        _QWORD *a8)
{
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // r10
  unsigned __int16 *v16; // r13
  unsigned __int16 v17; // ax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // eax
  unsigned __int16 *v23; // rax
  unsigned int v24; // edx
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned __int64 v27; // rsi
  int v28; // eax
  char v29; // di
  _QWORD *v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned int v33; // edx
  __int64 v34; // rdx
  __int128 v35; // [rsp+0h] [rbp-C0h]
  __int64 v36; // [rsp+10h] [rbp-B0h]
  __int64 v37; // [rsp+10h] [rbp-B0h]
  _QWORD v39[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v40; // [rsp+50h] [rbp-70h] BYREF
  int v41; // [rsp+58h] [rbp-68h]
  unsigned __int64 v42; // [rsp+60h] [rbp-60h]
  __int64 v43; // [rsp+68h] [rbp-58h]
  unsigned __int64 v44; // [rsp+70h] [rbp-50h] BYREF
  __int64 v45; // [rsp+78h] [rbp-48h]
  __int64 v46; // [rsp+80h] [rbp-40h]
  __int64 v47; // [rsp+88h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 80);
  v39[0] = a3;
  v39[1] = a4;
  v40 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v40, v11, 1);
  v41 = *(_DWORD *)(a2 + 72);
  if ( LOWORD(v39[0]) )
  {
    if ( LOWORD(v39[0]) == 1 || (unsigned __int16)(LOWORD(v39[0]) - 504) <= 7u )
      goto LABEL_33;
    v13 = *(_QWORD *)&byte_444C4A0[16 * LOWORD(v39[0]) - 16] >> 3;
    if ( (unsigned __int16)(LOWORD(v39[0]) - 176) <= 0x34u )
    {
LABEL_5:
      v14 = (unsigned int)v13;
      v15 = *(_QWORD *)(a1 + 8);
      v16 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a6 + 48LL) + 16LL * a6[2]);
      v17 = *v16;
      v18 = *((_QWORD *)v16 + 1);
      LOWORD(v44) = v17;
      v45 = v18;
      if ( !v17 )
      {
        v36 = v15;
        v19 = sub_3007260((__int64)&v44);
        v15 = v36;
        v46 = v19;
        v47 = v20;
LABEL_7:
        LODWORD(v45) = v19;
        if ( (unsigned int)v19 > 0x40 )
        {
          v37 = v15;
          sub_C43690((__int64)&v44, v14, 0);
          v15 = v37;
          v16 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a6 + 48LL) + 16LL * a6[2]);
        }
        else
        {
          v44 = v14;
        }
        *(_QWORD *)&v35 = sub_3401900(v15, (__int64)&v40, *v16, *((_QWORD *)v16 + 1), (__int64)&v44, 1, a7);
        *((_QWORD *)&v35 + 1) = v21;
        if ( (unsigned int)v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        v22 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
        *(_BYTE *)(a5 + 20) = 0;
        *(_QWORD *)a5 = 0;
        *(_QWORD *)(a5 + 8) = 0;
        *(_DWORD *)(a5 + 16) = v22;
        if ( a8 )
          *a8 += v14;
        v23 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a6 + 48LL) + 16LL * a6[2]);
        *(_QWORD *)a6 = sub_3405C90(
                          *(_QWORD **)(a1 + 8),
                          0x38u,
                          (__int64)&v40,
                          *v23,
                          *((_QWORD *)v23 + 1),
                          1,
                          a7,
                          *(_OWORD *)a6,
                          v35);
        a6[2] = v24;
        goto LABEL_21;
      }
      if ( v17 != 1 && (unsigned __int16)(v17 - 504) > 7u )
      {
        v19 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
        goto LABEL_7;
      }
LABEL_33:
      BUG();
    }
  }
  else
  {
    v42 = sub_3007260((__int64)v39);
    v43 = v12;
    v13 = v42 >> 3;
    if ( sub_3007100((__int64)v39) )
      goto LABEL_5;
  }
  v25 = *(_QWORD *)(a2 + 112);
  v26 = (unsigned int)v13 + *(_QWORD *)(v25 + 8);
  v27 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v27 )
  {
    v29 = *(_BYTE *)(v25 + 20);
    if ( (*(_QWORD *)v25 & 4) != 0 )
    {
      v28 = *(_DWORD *)(v27 + 12);
      v27 |= 4u;
    }
    else
    {
      v34 = *(_QWORD *)(v27 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 )
        v34 = **(_QWORD **)(v34 + 16);
      v28 = *(_DWORD *)(v34 + 8) >> 8;
    }
  }
  else
  {
    v28 = *(_DWORD *)(v25 + 16);
    v29 = 0;
  }
  *(_DWORD *)(a5 + 16) = v28;
  *(_QWORD *)(a5 + 8) = v26;
  *(_QWORD *)a5 = v27;
  *(_BYTE *)(a5 + 20) = v29;
  v30 = *(_QWORD **)(a1 + 8);
  v31 = *(_QWORD *)a6;
  v32 = *((_QWORD *)a6 + 1);
  LOBYTE(v47) = 0;
  v46 = (unsigned int)v13;
  *(_QWORD *)a6 = sub_3409320(v30, v31, v32, (unsigned int)v13, 0, (__int64)&v40, a7, 1);
  a6[2] = v33;
LABEL_21:
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
