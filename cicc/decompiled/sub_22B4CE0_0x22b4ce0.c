// Function: sub_22B4CE0
// Address: 0x22b4ce0
//
__int64 __fastcall sub_22B4CE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10,
        int a11,
        __int64 a12)
{
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 *v14; // rcx
  unsigned int v15; // eax
  __int64 *v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rdi
  char v19; // r10
  int v20; // eax
  __int64 v21; // rsi
  __int64 *v22; // r12
  int v23; // eax
  unsigned int v24; // edx
  __int64 *v25; // r9
  __int64 v26; // r11
  char v27; // al
  unsigned int v28; // r12d
  int v30; // r8d
  int v31; // r9d
  int v32; // r9d
  int v33; // r13d
  __int64 v34; // [rsp+0h] [rbp-60h] BYREF
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h] BYREF
  __int64 v39; // [rsp+28h] [rbp-38h]
  __int64 v40; // [rsp+30h] [rbp-30h]
  __int64 v41; // [rsp+38h] [rbp-28h]

  v34 = 0;
  v35 = 0;
  v12 = *(_QWORD *)(a7 + 16);
  v13 = *(_QWORD *)(a7 + 8);
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  sub_22B4A70(v13, v12, (__int64)&v34);
  sub_22B4A70(*(_QWORD *)(a10 + 8), *(_QWORD *)(a10 + 16), (__int64)&v38);
  v14 = (__int64 *)(v35 + 8LL * (unsigned int)v37);
  if ( !(_DWORD)v37 )
  {
LABEL_12:
    v18 = v39;
    v20 = v41;
    v21 = 8LL * (unsigned int)v41;
    v22 = (__int64 *)(v39 + v21);
    if ( !(_DWORD)v41 )
    {
      v28 = 1;
      goto LABEL_7;
    }
    v16 = (__int64 *)(v35 + 8LL * (unsigned int)v37);
    v19 = 0;
    goto LABEL_4;
  }
  v15 = (v37 - 1) & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
  v16 = (__int64 *)(v35 + 8LL * v15);
  v17 = *v16;
  if ( a9 != *v16 )
  {
    v30 = 1;
    while ( v17 != -4096 )
    {
      v32 = v30 + 1;
      v15 = (v37 - 1) & (v30 + v15);
      v16 = (__int64 *)(v35 + 8LL * v15);
      v17 = *v16;
      if ( a9 == *v16 )
        goto LABEL_3;
      v30 = v32;
    }
    goto LABEL_12;
  }
LABEL_3:
  v18 = v39;
  v19 = v14 != v16;
  v20 = v41;
  v21 = 8LL * (unsigned int)v41;
  v22 = (__int64 *)(v39 + v21);
  if ( !(_DWORD)v41 )
  {
LABEL_16:
    v27 = 0;
    goto LABEL_6;
  }
LABEL_4:
  v23 = v20 - 1;
  v24 = v23 & (((unsigned int)a12 >> 9) ^ ((unsigned int)a12 >> 4));
  v25 = (__int64 *)(v18 + 8LL * v24);
  v26 = *v25;
  if ( a12 != *v25 )
  {
    v31 = 1;
    while ( v26 != -4096 )
    {
      v33 = v31 + 1;
      v24 = v23 & (v31 + v24);
      v25 = (__int64 *)(v18 + 8LL * v24);
      v26 = *v25;
      if ( a12 == *v25 )
        goto LABEL_5;
      v31 = v33;
    }
    goto LABEL_16;
  }
LABEL_5:
  v27 = v22 != v25;
LABEL_6:
  v28 = 0;
  if ( v19 == v27 )
  {
    v28 = 1;
    if ( v14 != v16 )
      LOBYTE(v28) = a8 == a11;
  }
LABEL_7:
  sub_C7D6A0(v18, v21, 8);
  sub_C7D6A0(v35, 8LL * (unsigned int)v37, 8);
  return v28;
}
