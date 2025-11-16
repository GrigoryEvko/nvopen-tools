// Function: sub_F3D270
// Address: 0xf3d270
//
__int64 __fastcall sub_F3D270(__int64 a1, __int64 a2, const __m128i *a3)
{
  unsigned int v6; // ebx
  __m128i *v7; // r14
  int v8; // eax
  __int64 v9; // rdi
  __int64 v10; // rsi
  char v11; // al
  __int64 v12; // rax
  bool v13; // zf
  unsigned int v14; // ebx
  int v16; // eax
  bool v17; // al
  int v18; // [rsp+1Ch] [rbp-D4h]
  __m128i *v19; // [rsp+20h] [rbp-D0h]
  __int64 v20; // [rsp+30h] [rbp-C0h]
  unsigned int v21; // [rsp+38h] [rbp-B8h]
  int v22; // [rsp+4Ch] [rbp-A4h] BYREF
  __int64 v23; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v24; // [rsp+58h] [rbp-98h] BYREF
  __int64 v25[3]; // [rsp+60h] [rbp-90h] BYREF
  char v26; // [rsp+78h] [rbp-78h]
  __int64 v27; // [rsp+80h] [rbp-70h]
  _QWORD v28[3]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v29; // [rsp+A8h] [rbp-48h]
  __int64 v30; // [rsp+B0h] [rbp-40h]

  v6 = *(_DWORD *)(a2 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a2;
    v25[0] = 0;
LABEL_3:
    sub_F3D0B0(a2, 2 * v6);
    goto LABEL_4;
  }
  v12 = *(_QWORD *)(a2 + 8);
  v13 = a3[1].m128i_i8[8] == 0;
  v26 = 0;
  v25[0] = 0;
  v20 = v12;
  v27 = 0;
  memset(v28, 0, sizeof(v28));
  v29 = 1;
  v30 = 0;
  v22 = 0;
  if ( !v13 )
    v22 = a3[1].m128i_u16[0] | (a3->m128i_i32[2] << 16);
  v14 = v6 - 1;
  v24 = a3[2].m128i_i64[0];
  v23 = a3->m128i_i64[0];
  v18 = 1;
  v19 = 0;
  v21 = v14 & sub_F11290(&v23, &v22, &v24);
  while ( 1 )
  {
    v7 = (__m128i *)(v20 + 40LL * v21);
    if ( sub_F34140((__int64)a3, (__int64)v7) )
    {
      v9 = *(_QWORD *)a2;
      v10 = *(_QWORD *)(a2 + 8) + 40LL * *(unsigned int *)(a2 + 24);
      v11 = 0;
      goto LABEL_13;
    }
    if ( sub_F34140((__int64)v7, (__int64)v25) )
      break;
    v17 = sub_F34140((__int64)v7, (__int64)v28);
    if ( !v19 )
    {
      if ( !v17 )
        v7 = 0;
      v19 = v7;
    }
    v21 = v14 & (v18 + v21);
    ++v18;
  }
  v6 = *(_DWORD *)(a2 + 24);
  if ( v19 )
    v7 = v19;
  v16 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v8 = v16 + 1;
  v25[0] = (__int64)v7;
  if ( 4 * v8 >= 3 * v6 )
    goto LABEL_3;
  if ( v6 - (v8 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
    goto LABEL_5;
  sub_F3D0B0(a2, v6);
LABEL_4:
  sub_F38F60(a2, (__int64)a3, v25);
  v7 = (__m128i *)v25[0];
  v8 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
  *(_DWORD *)(a2 + 16) = v8;
  v28[0] = 0;
  LOBYTE(v29) = 0;
  v30 = 0;
  if ( !sub_F34140((__int64)v7, (__int64)v28) )
    --*(_DWORD *)(a2 + 20);
  *v7 = _mm_loadu_si128(a3);
  v7[1] = _mm_loadu_si128(a3 + 1);
  v7[2].m128i_i64[0] = a3[2].m128i_i64[0];
  v9 = *(_QWORD *)a2;
  v10 = *(_QWORD *)(a2 + 8) + 40LL * *(unsigned int *)(a2 + 24);
  v11 = 1;
LABEL_13:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v7;
  *(_BYTE *)(a1 + 32) = v11;
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 24) = v10;
  return a1;
}
