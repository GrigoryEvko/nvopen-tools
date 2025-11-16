// Function: sub_36DF750
// Address: 0x36df750
//
__int64 __fastcall sub_36DF750(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v6; // r9
  __int64 v7; // r14
  __int64 v10; // rsi
  unsigned __int64 *v11; // r15
  int v12; // eax
  int v13; // eax
  __int64 v14; // rsi
  unsigned __int8 *v15; // rax
  int v16; // edx
  int v17; // ecx
  unsigned __int8 *v18; // rsi
  __int64 v19; // rsi
  int v20; // edx
  __int64 v22; // rdx
  int v23; // eax
  unsigned int v24; // edx
  __int64 v25; // rax
  int v26; // eax
  unsigned int v27; // edx
  __int64 v28; // rax
  unsigned __int8 *v29; // rax
  int v30; // edx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // r8
  unsigned __int64 v33; // [rsp+8h] [rbp-D8h]
  unsigned int v34; // [rsp+10h] [rbp-D0h]
  unsigned int v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+18h] [rbp-C8h]
  int v37; // [rsp+18h] [rbp-C8h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+28h] [rbp-B8h]
  __int64 v41; // [rsp+60h] [rbp-80h] BYREF
  int v42; // [rsp+68h] [rbp-78h]
  unsigned __int64 v43; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-68h]
  unsigned __int64 v45; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v46; // [rsp+88h] [rbp-58h]
  unsigned __int64 v47; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v48; // [rsp+98h] [rbp-48h]
  unsigned __int64 v49; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v50; // [rsp+A8h] [rbp-38h]

  v6 = a2;
  v7 = a2;
  v10 = *(_QWORD *)(a2 + 80);
  v40 = *(_QWORD *)(a1 + 64);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 952) + 648LL) || (*(_BYTE *)(v6 + 32) & 1) == 0 )
  {
    v41 = v10;
    v11 = (unsigned __int64 *)&v41;
    if ( v10 )
    {
      v36 = v6;
      sub_B96E90((__int64)&v41, v10, 1);
      v6 = v36;
    }
    v12 = *(_DWORD *)(v6 + 72);
    v44 = 64;
    v43 = 0;
    v42 = v12;
    while ( 1 )
    {
      v13 = *(_DWORD *)(v7 + 24);
      if ( v13 != 56 && (v13 != 187 || (*(_BYTE *)(v7 + 28) & 8) == 0)
        || (v22 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL), v23 = *(_DWORD *)(v22 + 24), v23 != 35) && v23 != 11 )
      {
LABEL_8:
        if ( v44 <= 0x40 )
        {
          v14 = 0;
          if ( v44 )
            v14 = (__int64)(v43 << (64 - (unsigned __int8)v44)) >> (64 - (unsigned __int8)v44);
        }
        else
        {
          v14 = *(_QWORD *)v43;
        }
        v15 = sub_3401400(v40, v14, (__int64)&v41, 7, 0, 1u, a6, 0);
        v17 = v16;
        v18 = v15;
        if ( v44 > 0x40 && v43 )
        {
          v37 = v16;
          j_j___libc_free_0_0(v43);
          v17 = v37;
        }
        *(_QWORD *)a5 = v18;
        v19 = v41;
        *(_DWORD *)(a5 + 8) = v17;
        if ( v19 )
          goto LABEL_14;
        goto LABEL_15;
      }
      sub_C44830((__int64)&v45, (_DWORD *)(*(_QWORD *)(v22 + 96) + 24LL), 0x40u);
      v48 = v46;
      if ( v46 > 0x40 )
        sub_C43780((__int64)&v47, (const void **)&v45);
      else
        v47 = v45;
      sub_C45EE0((__int64)&v47, (__int64 *)&v43);
      v24 = v48;
      v48 = 0;
      v50 = v24;
      v25 = 1LL << ((unsigned __int8)v24 - 1);
      v49 = v47;
      if ( v24 <= 0x40 )
      {
        if ( (v25 & v47) != 0 )
        {
          if ( !v24 )
            goto LABEL_31;
          if ( v47 << (64 - (unsigned __int8)v24) == -1 )
          {
            v27 = v24 - 63;
          }
          else
          {
            _BitScanReverse64(&v31, ~(v47 << (64 - (unsigned __int8)v24)));
            v27 = v24 + 1 - (v31 ^ 0x3F);
          }
        }
        else
        {
          if ( !v47 )
            goto LABEL_31;
          _BitScanReverse64(&v32, v47);
          v27 = 65 - (v32 ^ 0x3F);
        }
      }
      else
      {
        v33 = v47;
        v34 = v24;
        if ( (*(_QWORD *)(v47 + 8LL * ((v24 - 1) >> 6)) & v25) != 0 )
          v26 = sub_C44500((__int64)&v49);
        else
          v26 = sub_C444A0((__int64)&v49);
        v27 = v34 + 1 - v26;
        if ( v33 )
        {
          v35 = v34 + 1 - v26;
          j_j___libc_free_0_0(v33);
          v27 = v35;
          if ( v48 > 0x40 )
          {
            if ( v47 )
            {
              j_j___libc_free_0_0(v47);
              v27 = v35;
            }
          }
        }
      }
      if ( v27 > 0x20 )
      {
        if ( v46 > 0x40 && v45 )
          j_j___libc_free_0_0(v45);
        goto LABEL_8;
      }
LABEL_31:
      sub_C45EE0((__int64)&v43, (__int64 *)&v45);
      v28 = *(_QWORD *)(v7 + 40);
      v7 = *(_QWORD *)v28;
      a3 = *(unsigned int *)(v28 + 8) | a3 & 0xFFFFFFFF00000000LL;
      if ( v46 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
    }
  }
  v49 = v10;
  v11 = &v49;
  if ( v10 )
  {
    v38 = v6;
    sub_B96E90((__int64)&v49, v10, 1);
    v6 = v38;
  }
  v50 = *(_DWORD *)(v6 + 72);
  v29 = sub_3400BD0(v40, 0, (__int64)&v49, 7, 0, 1u, a6, 0);
  v19 = v49;
  *(_QWORD *)a5 = v29;
  *(_DWORD *)(a5 + 8) = v30;
  if ( v19 )
LABEL_14:
    sub_B91220((__int64)v11, v19);
LABEL_15:
  *(_QWORD *)a4 = sub_36D7770(v7, a3, *(_QWORD **)(a1 + 64));
  *(_DWORD *)(a4 + 8) = v20;
  return 1;
}
