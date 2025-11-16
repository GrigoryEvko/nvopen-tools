// Function: sub_2122630
// Address: 0x2122630
//
__int64 __fastcall sub_2122630(__int64 *a1, __int64 a2, unsigned int a3, double a4, __m128i a5, __m128i a6)
{
  unsigned __int8 *v8; // rax
  __int64 v9; // r14
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // r9
  unsigned int v13; // r13d
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  char *v18; // rcx
  __m128i *v19; // rsi
  __int64 v20; // rdx
  char v21; // di
  int v22; // ecx
  __int64 v23; // rbx
  __int64 v25; // [rsp+8h] [rbp-88h]
  __int64 v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+10h] [rbp-80h] BYREF
  int v29; // [rsp+18h] [rbp-78h]
  _QWORD v30[4]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v31; // [rsp+40h] [rbp-50h] BYREF
  int v32; // [rsp+48h] [rbp-48h]
  __int64 v33; // [rsp+50h] [rbp-40h]

  v8 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v9 = *v8;
  v25 = *((_QWORD *)v8 + 1);
  sub_1F40D10((__int64)&v31, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v9, v25);
  v10 = *a1;
  if ( (_BYTE)v9 != (_BYTE)v32 )
    goto LABEL_2;
  if ( v25 == v33 )
  {
    if ( !(_BYTE)v9 )
      goto LABEL_2;
  }
  else if ( !(_BYTE)v9 )
  {
    goto LABEL_2;
  }
  if ( *(_QWORD *)(v10 + 8 * v9 + 120) )
    return a2;
LABEL_2:
  sub_1F40D10(
    (__int64)&v31,
    v10,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 72);
  v12 = v33;
  v13 = (unsigned __int8)v32;
  v28 = v11;
  if ( v11 )
  {
    v26 = v33;
    sub_1623A60((__int64)&v28, v11, 2);
    v12 = v26;
  }
  v14 = a1[1];
  v27 = v12;
  v29 = *(_DWORD *)(a2 + 64);
  v30[0] = sub_1D364E0(
             v14,
             (__int64)&v28,
             **(unsigned __int8 **)(a2 + 40),
             *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
             0,
             -0.0,
             *(double *)a5.m128i_i64,
             a6);
  v15 = *(_QWORD *)(a2 + 32);
  v30[1] = v16;
  v17 = sub_2120330((__int64)a1, *(_QWORD *)v15, *(_QWORD *)(v15 + 8));
  v18 = *(char **)(a2 + 40);
  v19 = (__m128i *)*a1;
  v30[2] = v17;
  v30[3] = v20;
  v21 = *v18;
  v22 = 57;
  if ( v21 != 9 )
  {
    v22 = 58;
    if ( v21 != 10 )
    {
      v22 = 59;
      if ( v21 != 11 )
      {
        v22 = 60;
        if ( v21 != 12 )
        {
          v22 = 462;
          if ( v21 == 13 )
            v22 = 61;
        }
      }
    }
  }
  sub_20BE530(
    (__int64)&v31,
    v19,
    a1[1],
    v22,
    v13,
    v27,
    (__m128i)0x8000000000000000LL,
    a5,
    a6,
    (__int64)v30,
    2u,
    0,
    (__int64)&v28,
    0,
    1);
  v23 = v31;
  if ( v28 )
    sub_161E7C0((__int64)&v28, v28);
  return v23;
}
