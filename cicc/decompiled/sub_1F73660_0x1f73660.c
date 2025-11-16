// Function: sub_1F73660
// Address: 0x1f73660
//
__int64 __fastcall sub_1F73660(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // r12
  int v9; // edx
  __int64 v10; // rax
  char v11; // r15
  unsigned __int64 v12; // r9
  int v13; // eax
  __int64 result; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  _DWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int8 v23; // r12
  int v24; // eax
  unsigned __int64 v25; // rdx
  __int64 v26; // rsi
  __int128 v27; // rax
  __int128 v28; // rax
  __int64 v29; // rcx
  unsigned int v30; // r15d
  unsigned int v31; // eax
  unsigned int v32; // eax
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  __int64 v36; // [rsp+10h] [rbp-80h]
  unsigned int v37; // [rsp+20h] [rbp-70h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  unsigned __int64 v39; // [rsp+28h] [rbp-68h]
  __int64 v40; // [rsp+28h] [rbp-68h]
  __int64 v41; // [rsp+28h] [rbp-68h]
  __int64 v42; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v43; // [rsp+38h] [rbp-58h]
  __int64 v44; // [rsp+40h] [rbp-50h] BYREF
  int v45; // [rsp+48h] [rbp-48h]
  unsigned __int8 v46[8]; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 v47; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(unsigned __int16 *)(a2 + 24);
  v8 = *(_QWORD *)v6;
  v9 = *(_DWORD *)(v6 + 8);
  v10 = *(_QWORD *)(v6 + 40);
  v11 = *(_BYTE *)(v10 + 88);
  v12 = *(_QWORD *)(v10 + 96);
  v13 = *(unsigned __int16 *)(v8 + 24);
  LOBYTE(v42) = v11;
  v43 = v12;
  if ( v13 == v7 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL);
    if ( v11 == *(_BYTE *)(v15 + 88) && (*(_QWORD *)(v15 + 96) == v12 || v11) )
      return v8;
  }
  if ( v13 != 145 )
    return 0;
  v38 = v12;
  if ( !sub_1D18C00(v8, 1, v9) )
    return 0;
  v17 = *(_DWORD **)(v8 + 32);
  v18 = v38;
  v19 = *(_QWORD *)v17;
  if ( v7 != *(unsigned __int16 *)(*(_QWORD *)v17 + 24LL) )
    return 0;
  v20 = *(_QWORD *)(a2 + 72);
  v37 = v17[2];
  v21 = *(_QWORD *)(v19 + 32);
  v44 = v20;
  v22 = *(_QWORD *)(v21 + 40);
  v23 = *(_BYTE *)(v22 + 88);
  v39 = *(_QWORD *)(v22 + 96);
  if ( v20 )
  {
    v33 = v19;
    v34 = v18;
    sub_1623A60((__int64)&v44, v20, 2);
    v19 = v33;
    v18 = v34;
  }
  v24 = *(_DWORD *)(a2 + 64);
  v46[0] = v23;
  v45 = v24;
  v47 = v39;
  if ( v11 == v23 )
  {
    if ( v23 || v39 == v18 )
    {
LABEL_15:
      v25 = v39;
      v26 = v23;
      goto LABEL_16;
    }
  }
  else if ( v11 )
  {
    v30 = sub_1F6C8D0(v11);
    goto LABEL_20;
  }
  v35 = v19;
  v32 = sub_1F58D40((__int64)&v42);
  v29 = v35;
  v30 = v32;
LABEL_20:
  if ( v23 )
  {
    v31 = sub_1F6C8D0(v23);
  }
  else
  {
    v36 = v29;
    v31 = sub_1F58D40((__int64)v46);
    v19 = v36;
  }
  if ( v31 <= v30 )
    goto LABEL_15;
  v26 = v42;
  v25 = v43;
LABEL_16:
  v40 = v19;
  *(_QWORD *)&v27 = sub_1D2EF30(*a1, v26, v25, v19, v16, v18);
  *(_QWORD *)&v28 = sub_1D332F0(
                      *a1,
                      v7,
                      (__int64)&v44,
                      *(unsigned __int8 *)(*(_QWORD *)(v40 + 40) + 16LL * v37),
                      *(const void ***)(*(_QWORD *)(v40 + 40) + 16LL * v37 + 8),
                      0,
                      a3,
                      a4,
                      a5,
                      **(_QWORD **)(v40 + 32),
                      *(_QWORD *)(*(_QWORD *)(v40 + 32) + 8LL),
                      v27);
  result = sub_1D309E0(
             *a1,
             145,
             (__int64)&v44,
             **(unsigned __int8 **)(a2 + 40),
             *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
             0,
             a3,
             a4,
             *(double *)a5.m128i_i64,
             v28);
  if ( v44 )
  {
    v41 = result;
    sub_161E7C0((__int64)&v44, v44);
    return v41;
  }
  return result;
}
