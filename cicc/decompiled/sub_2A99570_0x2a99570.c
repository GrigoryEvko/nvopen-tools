// Function: sub_2A99570
// Address: 0x2a99570
//
__int64 __fastcall sub_2A99570(__int64 a1, const __m128i *a2)
{
  char v4; // si
  __int32 v5; // edx
  __int32 v6; // ecx
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned int v11; // esi
  int v12; // eax
  __int64 v13; // r13
  int v14; // eax
  __int64 v15; // r8
  __int64 v16; // rdx
  __m128i v17; // xmm0
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // rcx
  __m128i *v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rdi
  _BYTE *v25; // rdi
  unsigned __int64 v26; // r15
  __int64 v27; // rdi
  char v28[4]; // [rsp+0h] [rbp-110h] BYREF
  __int32 v29; // [rsp+4h] [rbp-10Ch]
  __int32 v30; // [rsp+8h] [rbp-108h]
  __int64 v31; // [rsp+10h] [rbp-100h]
  int v32; // [rsp+18h] [rbp-F8h]
  _QWORD v33[2]; // [rsp+20h] [rbp-F0h] BYREF
  char v34; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v35; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+80h] [rbp-90h]
  _BYTE *v37; // [rsp+88h] [rbp-88h]
  __int64 v38; // [rsp+90h] [rbp-80h]
  _BYTE v39[120]; // [rsp+98h] [rbp-78h] BYREF

  v4 = a2->m128i_i8[0];
  v5 = a2->m128i_i32[2];
  v32 = 0;
  v6 = a2->m128i_i32[1];
  v7 = a2[1].m128i_i64[0];
  v28[0] = v4;
  v30 = v5;
  v29 = v6;
  v31 = v7;
  if ( (unsigned __int8)sub_2A92D80(a1, v28, v33) )
  {
    v9 = *(unsigned int *)(v33[0] + 24LL);
    return *(_QWORD *)(a1 + 32) + 104 * v9 + 24;
  }
  v11 = *(_DWORD *)(a1 + 24);
  v12 = *(_DWORD *)(a1 + 16);
  v13 = v33[0];
  ++*(_QWORD *)a1;
  v14 = v12 + 1;
  v15 = 2 * v11;
  v35.m128i_i64[0] = v13;
  if ( 4 * v14 >= 3 * v11 )
  {
    sub_2A99370(a1, v15);
  }
  else
  {
    if ( v11 - *(_DWORD *)(a1 + 20) - v14 > v11 >> 3 )
      goto LABEL_6;
    sub_2A99370(a1, v11);
  }
  sub_2A92D80(a1, v28, &v35);
  v13 = v35.m128i_i64[0];
  v14 = *(_DWORD *)(a1 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *(_QWORD *)(v13 + 16) != -4096
    || *(_DWORD *)(v13 + 8) != -1
    || *(_DWORD *)(v13 + 4) != -1
    || *(_BYTE *)v13 != 0xFF )
  {
    --*(_DWORD *)(a1 + 20);
  }
  *(_QWORD *)(v13 + 16) = v31;
  *(_DWORD *)(v13 + 8) = v30;
  *(_DWORD *)(v13 + 4) = v29;
  *(_BYTE *)v13 = v28[0];
  *(_DWORD *)(v13 + 24) = v32;
  v16 = a2[1].m128i_i64[0];
  v17 = _mm_loadu_si128(a2);
  v18 = *(unsigned int *)(a1 + 44);
  v33[0] = &v34;
  v36 = v16;
  v19 = *(unsigned int *)(a1 + 40);
  v33[1] = 0x800000000LL;
  v20 = v19 + 1;
  v38 = 0x800000000LL;
  v9 = v19;
  v37 = v39;
  v35 = v17;
  if ( v19 + 1 > v18 )
  {
    v26 = *(_QWORD *)(a1 + 32);
    v27 = a1 + 32;
    if ( v26 > (unsigned __int64)&v35 || (unsigned __int64)&v35 >= v26 + 104 * v19 )
    {
      sub_2A929B0(v27, v20, v19, v18, v15, v8);
      v19 = *(unsigned int *)(a1 + 40);
      v21 = *(_QWORD *)(a1 + 32);
      v22 = &v35;
      v9 = v19;
    }
    else
    {
      sub_2A929B0(v27, v20, v19, v18, v15, v8);
      v21 = *(_QWORD *)(a1 + 32);
      v19 = *(unsigned int *)(a1 + 40);
      v22 = (__m128i *)((char *)&v35 + v21 - v26);
      v9 = v19;
    }
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 32);
    v22 = &v35;
  }
  v23 = 13 * v19;
  v24 = v21 + 8 * v23;
  if ( v24 )
  {
    *(_BYTE *)v24 = v22->m128i_i8[0];
    *(_DWORD *)(v24 + 4) = v22->m128i_i32[1];
    *(_DWORD *)(v24 + 8) = v22->m128i_i32[2];
    *(_QWORD *)(v24 + 16) = v22[1].m128i_i64[0];
    *(_QWORD *)(v24 + 24) = v24 + 40;
    *(_QWORD *)(v24 + 32) = 0x800000000LL;
    if ( v22[2].m128i_i32[0] )
      sub_2A8A620(v24 + 24, (char **)&v22[1].m128i_i64[1], v23, v21, v15, v8);
    v9 = *(unsigned int *)(a1 + 40);
  }
  v25 = v37;
  *(_DWORD *)(a1 + 40) = v9 + 1;
  if ( v25 != v39 )
  {
    _libc_free((unsigned __int64)v25);
    v9 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v13 + 24) = v9;
  return *(_QWORD *)(a1 + 32) + 104 * v9 + 24;
}
