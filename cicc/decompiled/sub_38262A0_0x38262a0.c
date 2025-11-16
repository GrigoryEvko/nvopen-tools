// Function: sub_38262A0
// Address: 0x38262a0
//
void __fastcall sub_38262A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 v13; // rax
  int v14; // r9d
  __int64 v15; // rsi
  __int64 v16; // rdi
  int v17; // edx
  __int64 v18; // r15
  __int64 v19; // rax
  int v20; // edx
  unsigned __int16 v21; // ax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int8 *v26; // rsi
  unsigned int *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r9
  unsigned __int8 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  int v33; // edx
  int v34; // r9d
  unsigned __int8 *v35; // rax
  __int64 v36; // rsi
  int v37; // edx
  __int64 v38; // rdx
  __int128 v39; // [rsp-20h] [rbp-E0h]
  __int64 v40; // [rsp+0h] [rbp-C0h]
  __int64 (__fastcall *v41)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-B8h]
  __int64 v42; // [rsp+8h] [rbp-B8h]
  __int64 v43; // [rsp+8h] [rbp-B8h]
  unsigned int v44; // [rsp+40h] [rbp-80h] BYREF
  __int64 v45; // [rsp+48h] [rbp-78h]
  __int64 v46; // [rsp+50h] [rbp-70h] BYREF
  int v47; // [rsp+58h] [rbp-68h]
  __int64 v48; // [rsp+60h] [rbp-60h] BYREF
  char v49; // [rsp+68h] [rbp-58h]
  __int64 v50; // [rsp+70h] [rbp-50h] BYREF
  __int64 v51; // [rsp+78h] [rbp-48h]
  __int64 v52; // [rsp+80h] [rbp-40h]

  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v50, *a1, *(_QWORD *)(v13 + 64), v11, v12);
    LOWORD(v44) = v51;
    v45 = v52;
  }
  else
  {
    v44 = v9(*a1, *(_QWORD *)(v13 + 64), v11, v12);
    v45 = v38;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v46 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v46, v15, 1);
  v16 = a1[1];
  v47 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)a3 = sub_33FAF80(v16, 216, (__int64)&v46, v44, v45, v14, a5);
  *(_DWORD *)(a3 + 8) = v17;
  v18 = a1[1];
  v40 = *a1;
  v41 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
  v19 = sub_2E79000(*(__int64 **)(v18 + 40));
  if ( v41 == sub_2D42F30 )
  {
    v20 = sub_AE2980(v19, 0)[1];
    v21 = 2;
    if ( v20 != 1 )
    {
      v21 = 3;
      if ( v20 != 2 )
      {
        v21 = 4;
        if ( v20 != 4 )
        {
          v21 = 5;
          if ( v20 != 8 )
          {
            v21 = 6;
            if ( v20 != 16 )
            {
              v21 = 7;
              if ( v20 != 32 )
              {
                v21 = 8;
                if ( v20 != 64 )
                  v21 = 9 * (v20 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v21 = v41(v40, v19, 0);
  }
  v22 = v21;
  if ( (_WORD)v44 )
  {
    if ( (_WORD)v44 == 1 || (unsigned __int16)(v44 - 504) <= 7u )
      BUG();
    v24 = 16LL * ((unsigned __int16)v44 - 1);
    v23 = *(_QWORD *)&byte_444C4A0[v24];
    LOBYTE(v24) = byte_444C4A0[v24 + 8];
  }
  else
  {
    v42 = v21;
    v23 = sub_3007260((__int64)&v44);
    v22 = v42;
    v50 = v23;
    v51 = v24;
  }
  v43 = v22;
  v49 = v24;
  v48 = v23;
  v25 = sub_CA1930(&v48);
  v26 = sub_3400BD0(v18, v25, (__int64)&v46, v43, 0, 0, a5, 0);
  v27 = *(unsigned int **)(a2 + 40);
  *((_QWORD *)&v39 + 1) = v28;
  *(_QWORD *)&v39 = v26;
  v30 = sub_3406EB0(
          (_QWORD *)v18,
          0xC0u,
          (__int64)&v46,
          *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v27 + 48LL) + 16LL * v27[2]),
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v27 + 48LL) + 16LL * v27[2] + 8),
          v29,
          *(_OWORD *)v27,
          v39);
  v31 = v44;
  v32 = v45;
  *(_QWORD *)a4 = v30;
  *(_DWORD *)(a4 + 8) = v33;
  v35 = sub_33FAF80(a1[1], 216, (__int64)&v46, v31, v32, v34, a5);
  v36 = v46;
  *(_QWORD *)a4 = v35;
  *(_DWORD *)(a4 + 8) = v37;
  if ( v36 )
    sub_B91220((__int64)&v46, v36);
}
