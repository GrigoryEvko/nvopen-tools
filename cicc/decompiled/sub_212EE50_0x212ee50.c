// Function: sub_212EE50
// Address: 0x212ee50
//
unsigned __int64 __fastcall sub_212EE50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v11; // rsi
  __int64 v12; // rax
  char v13; // di
  __int64 v14; // rax
  __int64 v15; // rax
  char v16; // r8
  unsigned __int64 v17; // rax
  char v18; // r8
  unsigned int v19; // r15d
  __int64 v20; // rcx
  unsigned int v21; // esi
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r10
  unsigned int v25; // esi
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rcx
  __int128 v29; // rax
  unsigned int v30; // edx
  unsigned __int64 result; // rax
  __int128 v32; // rax
  int v33; // edx
  __int64 *v34; // r13
  __int64 v35; // rax
  unsigned int v36; // edx
  unsigned __int8 v37; // al
  __int128 v38; // rax
  unsigned int v39; // edx
  unsigned int v40; // eax
  __int64 v41; // [rsp+0h] [rbp-A0h]
  __int64 *v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+8h] [rbp-98h]
  __int64 *v44; // [rsp+8h] [rbp-98h]
  char v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+40h] [rbp-60h] BYREF
  int v47; // [rsp+48h] [rbp-58h]
  unsigned int v48; // [rsp+50h] [rbp-50h] BYREF
  const void **v49; // [rsp+58h] [rbp-48h]
  unsigned int v50; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v51; // [rsp+68h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 72);
  v46 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v46, v11, 2);
  v47 = *(_DWORD *)(a2 + 64);
  sub_20174B0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), (_DWORD *)a3, (_DWORD *)a4);
  v12 = *(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * *(unsigned int *)(a3 + 8);
  v13 = *(_BYTE *)v12;
  v49 = *(const void ***)(v12 + 8);
  v14 = *(_QWORD *)(a2 + 32);
  LOBYTE(v48) = v13;
  v15 = *(_QWORD *)(v14 + 40);
  v16 = *(_BYTE *)(v15 + 88);
  v17 = *(_QWORD *)(v15 + 96);
  LOBYTE(v50) = v16;
  v51 = v17;
  if ( v13 )
  {
    v19 = sub_2127930(v13);
  }
  else
  {
    v45 = v16;
    v40 = sub_1F58D40((__int64)&v48);
    v18 = v45;
    v19 = v40;
  }
  if ( v18 )
    v21 = sub_2127930(v18);
  else
    v21 = sub_1F58D40((__int64)&v50);
  v24 = *(_QWORD *)(a1 + 8);
  if ( v21 > v19 )
  {
    v25 = v21 - v19;
    if ( v25 == 32 )
    {
      LOBYTE(v26) = 5;
    }
    else if ( v25 > 0x20 )
    {
      if ( v25 == 64 )
      {
        LOBYTE(v26) = 6;
      }
      else
      {
        if ( v25 != 128 )
        {
LABEL_21:
          v43 = *(_QWORD *)(a1 + 8);
          v26 = sub_1F58CC0(*(_QWORD **)(v24 + 48), v25);
          v24 = v43;
          v41 = v26;
          goto LABEL_13;
        }
        LOBYTE(v26) = 7;
      }
    }
    else if ( v25 == 8 )
    {
      LOBYTE(v26) = 3;
    }
    else
    {
      LOBYTE(v26) = 4;
      if ( v25 != 16 )
      {
        LOBYTE(v26) = 2;
        if ( v25 != 1 )
          goto LABEL_21;
      }
    }
    v27 = 0;
LABEL_13:
    v28 = v41;
    v42 = (__int64 *)v24;
    LOBYTE(v28) = v26;
    *(_QWORD *)&v29 = sub_1D2EF30((_QWORD *)v24, (unsigned int)v28, v27, v28, v22, v23);
    *(_QWORD *)a4 = sub_1D332F0(
                      v42,
                      3,
                      (__int64)&v46,
                      v48,
                      v49,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      v29);
    result = v30;
    *(_DWORD *)(a4 + 8) = v30;
    goto LABEL_14;
  }
  v44 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v32 = sub_1D2EF30(v44, v50, v51, v20, v22, v23);
  *(_QWORD *)a3 = sub_1D332F0(
                    v44,
                    3,
                    (__int64)&v46,
                    v48,
                    v49,
                    0,
                    *(double *)a5.m128i_i64,
                    a6,
                    a7,
                    *(_QWORD *)a3,
                    *(_QWORD *)(a3 + 8),
                    v32);
  *(_DWORD *)(a3 + 8) = v33;
  v34 = *(__int64 **)(a1 + 8);
  v35 = sub_1E0A0C0(v34[4]);
  v36 = 8 * sub_15A9520(v35, 0);
  if ( v36 == 32 )
  {
    v37 = 5;
  }
  else if ( v36 > 0x20 )
  {
    v37 = 6;
    if ( v36 != 64 )
    {
      v37 = 0;
      if ( v36 == 128 )
        v37 = 7;
    }
  }
  else
  {
    v37 = 3;
    if ( v36 != 8 )
      v37 = 4 * (v36 == 16);
  }
  *(_QWORD *)&v38 = sub_1D38BB0((__int64)v34, v19 - 1, (__int64)&v46, v37, 0, 0, a5, a6, a7, 0);
  *(_QWORD *)a4 = sub_1D332F0(
                    v34,
                    123,
                    (__int64)&v46,
                    v48,
                    v49,
                    0,
                    *(double *)a5.m128i_i64,
                    a6,
                    a7,
                    *(_QWORD *)a3,
                    *(_QWORD *)(a3 + 8),
                    v38);
  result = v39;
  *(_DWORD *)(a4 + 8) = v39;
LABEL_14:
  if ( v46 )
    return sub_161E7C0((__int64)&v46, v46);
  return result;
}
