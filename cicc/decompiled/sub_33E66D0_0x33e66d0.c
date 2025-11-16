// Function: sub_33E66D0
// Address: 0x33e66d0
//
__int64 __fastcall sub_33E66D0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8)
{
  int v8; // esi
  int v9; // r9d
  __int64 v12; // r11
  __int64 v13; // r10
  _QWORD *v14; // rsi
  __int64 v15; // r14
  char v16; // r15
  int v17; // r8d
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-F8h]
  __int64 v25; // [rsp+8h] [rbp-F8h]
  int v26; // [rsp+10h] [rbp-F0h]
  __int64 v27; // [rsp+10h] [rbp-F0h]
  __int64 v28; // [rsp+10h] [rbp-F0h]
  int v29; // [rsp+18h] [rbp-E8h]
  int v30; // [rsp+18h] [rbp-E8h]
  int v31; // [rsp+20h] [rbp-E0h]
  __int64 v32; // [rsp+20h] [rbp-E0h]
  int v33; // [rsp+20h] [rbp-E0h]
  unsigned int v34; // [rsp+28h] [rbp-D8h]
  __int64 v35; // [rsp+28h] [rbp-D8h]
  __int64 *v36; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v37[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v38[176]; // [rsp+50h] [rbp-B0h] BYREF

  v8 = ~a2;
  v9 = a5;
  v12 = (__int64)a7;
  v13 = a8;
  v34 = v8;
  if ( *(_WORD *)(a4 + 16LL * (unsigned int)(a5 - 1)) == 262 )
  {
    v36 = 0;
    v16 = 0;
  }
  else
  {
    v26 = a5;
    v37[1] = 0x2000000000LL;
    v37[0] = (unsigned __int64)v38;
    sub_33C9670((__int64)v37, v8, a4, a7, a8, a5);
    v36 = 0;
    v14 = sub_33CCCF0((__int64)a1, (__int64)v37, a3, (__int64 *)&v36);
    if ( v14 )
    {
      v15 = sub_33CEC90((__int64)a1, (__int64)v14, a3);
      if ( (_BYTE *)v37[0] != v38 )
        _libc_free(v37[0]);
      return v15;
    }
    v12 = (__int64)a7;
    v13 = a8;
    v9 = v26;
    if ( (_BYTE *)v37[0] != v38 )
    {
      _libc_free(v37[0]);
      v13 = a8;
      v12 = (__int64)a7;
      v9 = v26;
    }
    v16 = 1;
  }
  v15 = a1[52];
  v17 = *(_DWORD *)(a3 + 8);
  if ( v15 )
  {
    a1[52] = *(_QWORD *)v15;
LABEL_8:
    v18 = *(_QWORD *)a3;
    v37[0] = v18;
    if ( v18 )
    {
      v24 = v13;
      v27 = v12;
      v29 = v9;
      v31 = v17;
      sub_B96E90((__int64)v37, v18, 1);
      v13 = v24;
      v12 = v27;
      v9 = v29;
      v17 = v31;
    }
    *(_QWORD *)v15 = 0;
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = 0;
    *(_QWORD *)(v15 + 24) = v34;
    *(_WORD *)(v15 + 34) = -1;
    *(_DWORD *)(v15 + 36) = -1;
    *(_QWORD *)(v15 + 40) = 0;
    *(_QWORD *)(v15 + 48) = a4;
    *(_QWORD *)(v15 + 56) = 0;
    *(_DWORD *)(v15 + 64) = 0;
    *(_DWORD *)(v15 + 68) = v9;
    *(_DWORD *)(v15 + 72) = v17;
    v19 = (unsigned __int8 *)v37[0];
    *(_QWORD *)(v15 + 80) = v37[0];
    if ( v19 )
    {
      v32 = v13;
      v35 = v12;
      sub_B976B0((__int64)v37, v19, v15 + 80);
      v12 = v35;
      *(_QWORD *)(v15 + 88) = 0xFFFFFFFFLL;
      v13 = v32;
    }
    else
    {
      *(_QWORD *)(v15 + 88) = 0xFFFFFFFFLL;
    }
    *(_WORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 96) = 0;
    *(_DWORD *)(v15 + 104) = 0;
    goto LABEL_13;
  }
  v21 = a1[53];
  a1[63] += 120LL;
  v22 = (v21 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v22 + 120 || !v21 )
  {
    v25 = v13;
    v28 = v12;
    v30 = v9;
    v33 = v17;
    v23 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    v13 = v25;
    v12 = v28;
    v9 = v30;
    v17 = v33;
    v15 = v23;
    goto LABEL_8;
  }
  a1[53] = v22 + 120;
  if ( v22 )
  {
    v15 = (v21 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_8;
  }
LABEL_13:
  sub_33E4EC0((__int64)a1, v15, v12, v13);
  if ( v16 )
    sub_C657C0(a1 + 65, (__int64 *)v15, v36, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v15);
  return v15;
}
