// Function: sub_20A2230
// Address: 0x20a2230
//
__int64 __fastcall sub_20A2230(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  unsigned int v10; // r13d
  __int64 v13; // rsi
  int v14; // r10d
  __int64 (*v15)(); // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 *v19; // r8
  char v20; // al
  char v21; // al
  char v22; // al
  unsigned int v23; // r11d
  unsigned int v24; // eax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int128 v27; // rax
  unsigned int v28; // r11d
  unsigned int v29; // r10d
  int v30; // edx
  unsigned int v31; // [rsp+0h] [rbp-C0h]
  __int64 *v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  __int64 v34; // [rsp+10h] [rbp-B0h]
  int v35; // [rsp+10h] [rbp-B0h]
  __int128 v36; // [rsp+10h] [rbp-B0h]
  __int128 v37; // [rsp+10h] [rbp-B0h]
  int v38; // [rsp+10h] [rbp-B0h]
  unsigned int v39; // [rsp+20h] [rbp-A0h]
  unsigned int v40; // [rsp+20h] [rbp-A0h]
  __int64 v41; // [rsp+20h] [rbp-A0h]
  int v42; // [rsp+28h] [rbp-98h]
  unsigned int v43; // [rsp+28h] [rbp-98h]
  unsigned int v44; // [rsp+28h] [rbp-98h]
  unsigned int v45; // [rsp+28h] [rbp-98h]
  __int64 *v46; // [rsp+30h] [rbp-90h]
  int v47; // [rsp+30h] [rbp-90h]
  __int64 *v48; // [rsp+30h] [rbp-90h]
  const void **v49; // [rsp+30h] [rbp-90h]
  __int64 *v50; // [rsp+38h] [rbp-88h]
  __int64 v51; // [rsp+60h] [rbp-60h] BYREF
  int v52; // [rsp+68h] [rbp-58h]
  __int64 v53; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+78h] [rbp-48h]
  __int64 v55; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v56; // [rsp+88h] [rbp-38h]

  v10 = a1;
  v13 = *(_QWORD *)(a2 + 72);
  v50 = *(__int64 **)a5;
  v51 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v51, v13, 2);
  v14 = *(unsigned __int16 *)(a2 + 24);
  v52 = *(_DWORD *)(a2 + 64);
  v15 = *(__int64 (**)())(*(_QWORD *)a1 + 1056LL);
  if ( v15 != sub_20A07D0 )
  {
    v47 = v14;
    v21 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v15)(a1, a2, a3, a4, a5);
    v14 = v47;
    if ( v21 )
    {
      LOBYTE(v10) = *(_QWORD *)(a5 + 32) != 0;
      goto LABEL_5;
    }
  }
  v10 = 0;
  if ( (unsigned int)(v14 - 118) <= 2 )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
    LOBYTE(v10) = *(_WORD *)(v17 + 24) == 32 || *(_WORD *)(v17 + 24) == 10;
    if ( (_BYTE)v10 )
    {
      v18 = *(_QWORD *)(v17 + 88);
      v19 = (__int64 *)(v18 + 24);
      if ( v14 == 120 )
      {
        if ( *(_DWORD *)(a4 + 8) <= 0x40u )
        {
          if ( (*(_QWORD *)a4 & ~*(_QWORD *)(v18 + 24)) == 0 )
            goto LABEL_12;
        }
        else
        {
          v34 = v18;
          v48 = (__int64 *)(v18 + 24);
          v22 = sub_16A5A00((__int64 *)a4, (__int64 *)(v18 + 24));
          v19 = v48;
          v14 = 120;
          v18 = v34;
          if ( v22 )
            goto LABEL_12;
        }
      }
      if ( *(_DWORD *)(v18 + 32) <= 0x40u )
      {
        if ( (*(_QWORD *)(v18 + 24) & ~*(_QWORD *)a4) == 0 )
          goto LABEL_12;
      }
      else
      {
        v33 = v18;
        v42 = v14;
        v46 = v19;
        v20 = sub_16A5A00(v19, (__int64 *)a4);
        v19 = v46;
        v14 = v42;
        v18 = v33;
        if ( v20 )
        {
LABEL_12:
          v10 = 0;
          goto LABEL_5;
        }
      }
      v23 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
      v24 = *(_DWORD *)(a4 + 8);
      v49 = *(const void ***)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3 + 8);
      v54 = v24;
      if ( v24 > 0x40 )
      {
        v31 = v23;
        v32 = v19;
        v41 = v18;
        v38 = v14;
        sub_16A4FD0((__int64)&v53, (const void **)a4);
        v24 = v54;
        v14 = v38;
        v18 = v41;
        v23 = v31;
        if ( v54 > 0x40 )
        {
          sub_16A8890(&v53, v32);
          v24 = v54;
          v26 = v53;
          v23 = v31;
          v14 = v38;
LABEL_22:
          v55 = v26;
          v35 = v14;
          v43 = v23;
          v56 = v24;
          v54 = 0;
          *(_QWORD *)&v27 = sub_1D38970((__int64)v50, (__int64)&v55, (__int64)&v51, v23, v49, 0, a6, a7, a8, 0);
          v28 = v43;
          v29 = v35;
          if ( v56 > 0x40 && v55 )
          {
            v39 = v43;
            v36 = v27;
            v44 = v29;
            j_j___libc_free_0_0(v55);
            v28 = v39;
            v27 = v36;
            v29 = v44;
          }
          if ( v54 > 0x40 && v53 )
          {
            v40 = v28;
            v37 = v27;
            v45 = v29;
            j_j___libc_free_0_0(v53);
            v28 = v40;
            v27 = v37;
            v29 = v45;
          }
          *(_QWORD *)(a5 + 32) = sub_1D332F0(
                                   v50,
                                   v29,
                                   (__int64)&v51,
                                   v28,
                                   v49,
                                   0,
                                   *(double *)a6.m128i_i64,
                                   a7,
                                   a8,
                                   **(_QWORD **)(a2 + 32),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                                   v27);
          *(_QWORD *)(a5 + 16) = a2;
          *(_DWORD *)(a5 + 24) = a3;
          *(_DWORD *)(a5 + 40) = v30;
          goto LABEL_5;
        }
        v25 = v53;
      }
      else
      {
        v25 = *(_QWORD *)a4;
      }
      v53 = *(_QWORD *)(v18 + 24) & v25;
      v26 = v53;
      goto LABEL_22;
    }
  }
LABEL_5:
  if ( v51 )
    sub_161E7C0((__int64)&v51, v51);
  return v10;
}
