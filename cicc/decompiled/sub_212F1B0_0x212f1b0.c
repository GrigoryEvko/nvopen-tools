// Function: sub_212F1B0
// Address: 0x212f1b0
//
unsigned __int64 __fastcall sub_212F1B0(
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
  unsigned int v18; // edx
  char v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rcx
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
  __int64 *v33; // rax
  __int64 v34; // rcx
  const void **v35; // r8
  int v36; // edx
  int v37; // edx
  unsigned int v38; // eax
  unsigned __int64 v39; // [rsp-10h] [rbp-B0h]
  __int64 v40; // [rsp+0h] [rbp-A0h]
  unsigned int v41; // [rsp+8h] [rbp-98h]
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
    v18 = sub_2127930(v13);
  }
  else
  {
    v45 = v16;
    v38 = sub_1F58D40((__int64)&v48);
    v19 = v45;
    v18 = v38;
  }
  v41 = v18;
  if ( v19 )
    v20 = sub_2127930(v19);
  else
    v20 = sub_1F58D40((__int64)&v50);
  v24 = *(_QWORD *)(a1 + 8);
  if ( v20 > v41 )
  {
    v25 = v20 - v41;
    if ( v20 - v41 == 32 )
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
LABEL_18:
          v43 = *(_QWORD *)(a1 + 8);
          v26 = sub_1F58CC0(*(_QWORD **)(v24 + 48), v25);
          v24 = v43;
          v40 = v26;
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
          goto LABEL_18;
      }
    }
    v27 = 0;
LABEL_13:
    v28 = v40;
    v42 = (__int64 *)v24;
    LOBYTE(v28) = v26;
    *(_QWORD *)&v29 = sub_1D2EF30((_QWORD *)v24, (unsigned int)v28, v27, v28, v22, v23);
    *(_QWORD *)a4 = sub_1D332F0(
                      v42,
                      4,
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
    goto LABEL_20;
  }
  v44 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v32 = sub_1D2EF30(v44, v50, v51, v21, v22, v23);
  v33 = sub_1D332F0(
          v44,
          4,
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
  v34 = v48;
  v35 = v49;
  *(_QWORD *)a3 = v33;
  *(_DWORD *)(a3 + 8) = v36;
  *(_QWORD *)a4 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v46, v34, v35, 0, a5, a6, a7, 0);
  *(_DWORD *)(a4 + 8) = v37;
  result = v39;
LABEL_20:
  if ( v46 )
    return sub_161E7C0((__int64)&v46, v46);
  return result;
}
