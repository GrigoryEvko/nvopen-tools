// Function: sub_3832280
// Address: 0x3832280
//
unsigned __int8 *__fastcall sub_3832280(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned int *v4; // rdx
  __int64 v5; // rax
  unsigned __int16 v6; // cx
  __int64 v7; // rax
  unsigned __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rcx
  unsigned int v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r8
  __int64 v22; // r15
  __int64 v23; // rax
  int v24; // edx
  unsigned __int16 v25; // ax
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int128 v30; // rax
  __int64 v31; // r9
  unsigned int v32; // edx
  __int64 v33; // r9
  unsigned __int8 *v34; // r12
  __int128 v36; // [rsp-30h] [rbp-D0h]
  __int128 v37; // [rsp-10h] [rbp-B0h]
  __int64 v38; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v39; // [rsp+8h] [rbp-98h]
  __int64 (__fastcall *v40)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-98h]
  __int64 v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+10h] [rbp-90h]
  __int128 v44; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v45; // [rsp+20h] [rbp-80h]
  unsigned __int16 v46; // [rsp+30h] [rbp-70h] BYREF
  __int64 v47; // [rsp+38h] [rbp-68h]
  __int64 v48; // [rsp+40h] [rbp-60h] BYREF
  int v49; // [rsp+48h] [rbp-58h]
  __int64 v50; // [rsp+50h] [rbp-50h] BYREF
  char v51; // [rsp+58h] [rbp-48h]
  __int64 v52; // [rsp+60h] [rbp-40h] BYREF
  __int64 v53; // [rsp+68h] [rbp-38h]

  v4 = *(unsigned int **)(a2 + 40);
  v5 = *(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * v4[2];
  v6 = *(_WORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v46 = v6;
  v47 = v7;
  v8 = *(_QWORD *)v4;
  v9 = *((_QWORD *)v4 + 1);
  v10 = *(_QWORD *)(*(_QWORD *)v4 + 80LL);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * v4[2] + 8);
  v12 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * v4[2]);
  v52 = v10;
  v43 = v11;
  if ( v10 )
  {
    v39 = v8;
    sub_B96E90((__int64)&v52, v10, 1);
    v8 = v39;
  }
  LODWORD(v53) = *(_DWORD *)(v8 + 72);
  v13 = sub_37AE0F0(a1, v8, v9);
  *(_QWORD *)&v44 = sub_34070B0(*(_QWORD **)(a1 + 8), v13, v9 & 0xFFFFFFFF00000000LL | v14, (__int64)&v52, v12, v43, a3);
  *((_QWORD *)&v44 + 1) = v15;
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
  v16 = sub_37AE0F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v17 = *(_QWORD *)(a2 + 80);
  v18 = v16;
  v20 = v19;
  v48 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v48, v17, 1);
  v21 = *(_QWORD *)a1;
  v22 = *(_QWORD *)(a1 + 8);
  v49 = *(_DWORD *)(a2 + 72);
  v38 = v21;
  v40 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v21 + 32LL);
  v23 = sub_2E79000(*(__int64 **)(v22 + 40));
  if ( v40 == sub_2D42F30 )
  {
    v24 = sub_AE2980(v23, 0)[1];
    v25 = 2;
    if ( v24 != 1 )
    {
      v25 = 3;
      if ( v24 != 2 )
      {
        v25 = 4;
        if ( v24 != 4 )
        {
          v25 = 5;
          if ( v24 != 8 )
          {
            v25 = 6;
            if ( v24 != 16 )
            {
              v25 = 7;
              if ( v24 != 32 )
              {
                v25 = 8;
                if ( v24 != 64 )
                  v25 = 9 * (v24 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v25 = v40(v38, v23, 0);
  }
  v26 = v25;
  if ( v46 )
  {
    if ( v46 == 1 || (unsigned __int16)(v46 - 504) <= 7u )
      BUG();
    v28 = 16LL * (v46 - 1);
    v27 = *(_QWORD *)&byte_444C4A0[v28];
    LOBYTE(v28) = byte_444C4A0[v28 + 8];
  }
  else
  {
    v41 = v25;
    v27 = sub_3007260((__int64)&v46);
    v26 = v41;
    v52 = v27;
    v53 = v28;
  }
  v42 = v26;
  v51 = v28;
  v50 = v27;
  v29 = sub_CA1930(&v50);
  *(_QWORD *)&v30 = sub_3400BD0(v22, v29, (__int64)&v48, v42, 0, 0, a3, 0);
  *((_QWORD *)&v36 + 1) = v20;
  *(_QWORD *)&v36 = v18;
  v45 = sub_3406EB0(
          (_QWORD *)v22,
          0xBEu,
          (__int64)&v48,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v31,
          v36,
          v30);
  *((_QWORD *)&v37 + 1) = v32 | v20 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v37 = v45;
  v34 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          0xBBu,
          (__int64)&v48,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v33,
          v44,
          v37);
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v34;
}
