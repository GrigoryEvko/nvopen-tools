// Function: sub_315E720
// Address: 0x315e720
//
__int64 __fastcall sub_315E720(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r13
  const char *v10; // rax
  __int64 v11; // rdx
  __m128i v12; // rax
  char v13; // cl
  _QWORD *v14; // rsi
  char v15; // cl
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  _QWORD *v19; // r10
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 result; // rax
  __m128i v31; // xmm1
  _QWORD *v32; // rsi
  __int64 v35; // [rsp+20h] [rbp-120h]
  __int64 v36; // [rsp+28h] [rbp-118h]
  __int64 v37; // [rsp+30h] [rbp-110h]
  __int64 v38; // [rsp+40h] [rbp-100h]
  _QWORD *v39; // [rsp+40h] [rbp-100h]
  __int64 v40; // [rsp+48h] [rbp-F8h]
  _QWORD v41[4]; // [rsp+50h] [rbp-F0h] BYREF
  char v42; // [rsp+70h] [rbp-D0h]
  char v43; // [rsp+71h] [rbp-CFh]
  __m128i v44; // [rsp+80h] [rbp-C0h] BYREF
  __m128i v45; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+A0h] [rbp-A0h]
  const char *v47; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v48; // [rsp+B8h] [rbp-88h]
  __int16 v49; // [rsp+D0h] [rbp-70h]
  __m128i v50; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v51; // [rsp+F0h] [rbp-50h]
  __int64 v52; // [rsp+100h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 56);
  if ( v5 )
    v5 -= 24;
  do
  {
    v6 = *(_QWORD *)(v5 - 8);
    v7 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) != 0 )
    {
      v8 = 0;
      do
      {
        if ( a2 == *(_QWORD *)(v6 + 32LL * *(unsigned int *)(v5 + 72) + 8 * v8) )
        {
          v7 = 32 * v8;
          goto LABEL_8;
        }
        ++v8;
      }
      while ( (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) != (_DWORD)v8 );
      v7 = 0x1FFFFFFFE0LL;
    }
LABEL_8:
    v9 = *(_QWORD *)(v6 + v7);
    v10 = sub_BD5D20(a1);
    v43 = 1;
    v49 = 261;
    v47 = v10;
    v48 = v11;
    v41[0] = ".";
    v42 = 3;
    v12.m128i_i64[0] = (__int64)sub_BD5D20(v9);
    v13 = v42;
    if ( !v42 )
    {
      LOWORD(v46) = 256;
LABEL_13:
      LOWORD(v52) = 256;
      goto LABEL_14;
    }
    if ( v42 == 1 )
    {
      v44 = v12;
      LOWORD(v46) = 261;
      v15 = v49;
      if ( !(_BYTE)v49 )
        goto LABEL_13;
      if ( (_BYTE)v49 == 1 )
        goto LABEL_38;
      v36 = v12.m128i_i64[1];
      v12.m128i_i8[8] = 5;
      if ( HIBYTE(v49) == 1 )
        goto LABEL_44;
    }
    else
    {
      if ( v43 == 1 )
      {
        v14 = (_QWORD *)v41[0];
        v37 = v41[1];
      }
      else
      {
        v14 = v41;
        v13 = 2;
      }
      v44 = v12;
      BYTE1(v46) = v13;
      v15 = v49;
      v45.m128i_i64[0] = (__int64)v14;
      v45.m128i_i64[1] = v37;
      LOBYTE(v46) = 5;
      if ( !(_BYTE)v49 )
        goto LABEL_13;
      if ( (_BYTE)v49 == 1 )
      {
LABEL_38:
        v31 = _mm_loadu_si128(&v45);
        v50 = _mm_loadu_si128(&v44);
        v52 = v46;
        v51 = v31;
        goto LABEL_14;
      }
      v12.m128i_i64[0] = (__int64)&v44;
      v12.m128i_i8[8] = 2;
      if ( HIBYTE(v49) == 1 )
      {
LABEL_44:
        v32 = v47;
        v35 = v48;
        goto LABEL_42;
      }
    }
    v32 = &v47;
    v15 = 2;
LABEL_42:
    v50.m128i_i64[0] = v12.m128i_i64[0];
    v51.m128i_i64[0] = (__int64)v32;
    v50.m128i_i64[1] = v36;
    LOBYTE(v52) = v12.m128i_i8[8];
    v51.m128i_i64[1] = v35;
    BYTE1(v52) = v15;
LABEL_14:
    v38 = *(_QWORD *)(v9 + 8);
    v16 = sub_BD2DA0(80);
    v17 = v38;
    v18 = v16;
    if ( v16 )
    {
      v39 = (_QWORD *)v16;
      sub_B44260(v16, v17, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v18 + 72) = 1;
      sub_BD6B50((unsigned __int8 *)v18, (const char **)&v50);
      sub_BD2A10(v18, *(_DWORD *)(v18 + 72), 1);
      v19 = v39;
    }
    else
    {
      v19 = 0;
    }
    v20 = v40;
    LOWORD(v20) = 1;
    v40 = v20;
    sub_B44220(v19, *(_QWORD *)(a2 + 56), v20);
    v21 = *(_DWORD *)(v18 + 4) & 0x7FFFFFF;
    if ( v21 == *(_DWORD *)(v18 + 72) )
    {
      sub_B48D90(v18);
      v21 = *(_DWORD *)(v18 + 4) & 0x7FFFFFF;
    }
    v22 = (v21 + 1) & 0x7FFFFFF;
    v23 = v22 | *(_DWORD *)(v18 + 4) & 0xF8000000;
    v24 = *(_QWORD *)(v18 - 8) + 32LL * (unsigned int)(v22 - 1);
    *(_DWORD *)(v18 + 4) = v23;
    if ( *(_QWORD *)v24 )
    {
      v25 = *(_QWORD *)(v24 + 8);
      **(_QWORD **)(v24 + 16) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
    }
    *(_QWORD *)v24 = v9;
    v26 = *(_QWORD *)(v9 + 16);
    *(_QWORD *)(v24 + 8) = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = v24 + 8;
    *(_QWORD *)(v24 + 16) = v9 + 16;
    *(_QWORD *)(v9 + 16) = v24;
    *(_QWORD *)(*(_QWORD *)(v18 - 8)
              + 32LL * *(unsigned int *)(v18 + 72)
              + 8LL * ((*(_DWORD *)(v18 + 4) & 0x7FFFFFFu) - 1)) = a3;
    v27 = *(_QWORD *)(v5 - 8) + v7;
    if ( *(_QWORD *)v27 )
    {
      v28 = *(_QWORD *)(v27 + 8);
      **(_QWORD **)(v27 + 16) = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 16) = *(_QWORD *)(v27 + 16);
    }
    *(_QWORD *)v27 = v18;
    v29 = *(_QWORD *)(v18 + 16);
    *(_QWORD *)(v27 + 8) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = v27 + 8;
    *(_QWORD *)(v27 + 16) = v18 + 16;
    *(_QWORD *)(v18 + 16) = v27;
    result = *(_QWORD *)(v5 + 32);
    if ( result == *(_QWORD *)(v5 + 40) + 48LL || !result )
      BUG();
    v5 = 0;
    if ( *(_BYTE *)(result - 24) == 84 )
      v5 = result - 24;
  }
  while ( v5 != a4 );
  return result;
}
