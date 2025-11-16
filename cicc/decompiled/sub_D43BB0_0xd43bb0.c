// Function: sub_D43BB0
// Address: 0xd43bb0
//
__int64 __fastcall sub_D43BB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r12
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rsi
  _QWORD *v22; // r12
  _QWORD *v23; // r14
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // rdi
  _QWORD *v32; // r14
  _QWORD *v33; // r12
  __int64 v34; // rax
  __int64 result; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  int v40; // [rsp+1Ch] [rbp-54h]
  __int64 v41; // [rsp+20h] [rbp-50h] BYREF
  char v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+30h] [rbp-40h]
  __int64 v44; // [rsp+38h] [rbp-38h]

  v10 = a2;
  v12 = sub_22077B0(168);
  v13 = v12;
  if ( v12 )
  {
    v14 = a2;
    a2 = a3;
    sub_D9AF80(v12, a3, v14);
  }
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x600000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v10;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  v40 = -1;
  if ( a4 )
  {
    if ( !(unsigned __int8)sub_DFE640(a4) )
    {
      a2 = 1;
      v43 = sub_DFB1B0(a4, 1);
      v44 = v37;
      v41 = 2 * v43;
      v42 = v37;
      v40 = sub_CA1930(&v41);
    }
    v13 = *(_QWORD *)a1;
  }
  v15 = sub_22077B0(448);
  v16 = v15;
  if ( v15 )
  {
    *(_QWORD *)v15 = v13;
    *(_QWORD *)(v15 + 8) = v10;
    *(_QWORD *)(v15 + 24) = 0;
    *(_QWORD *)(v15 + 16) = a1 + 120;
    *(_QWORD *)(v15 + 56) = v15 + 72;
    *(_QWORD *)(v15 + 64) = 0x1000000000LL;
    *(_QWORD *)(v15 + 216) = 0xFFFFFFFFLL;
    *(_QWORD *)(v15 + 240) = v15 + 256;
    *(_QWORD *)(v15 + 248) = 0x800000000LL;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 40) = 0;
    *(_DWORD *)(v15 + 48) = 0;
    *(_DWORD *)(v15 + 200) = 0;
    *(_QWORD *)(v15 + 208) = 0;
    *(_BYTE *)(v15 + 224) = 0;
    *(_DWORD *)(v15 + 228) = 0;
    *(_BYTE *)(v15 + 232) = 1;
    *(_DWORD *)(v15 + 352) = v40;
    *(_QWORD *)(v15 + 360) = 0;
    *(_QWORD *)(v15 + 368) = 0;
    *(_QWORD *)(v15 + 376) = 0;
    *(_DWORD *)(v15 + 384) = 0;
    *(_BYTE *)(v15 + 440) = 0;
  }
  v17 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = v15;
  if ( v17 )
  {
    if ( *(_BYTE *)(v17 + 440) )
    {
      v36 = *(unsigned int *)(v17 + 416);
      *(_BYTE *)(v17 + 440) = 0;
      sub_C7D6A0(*(_QWORD *)(v17 + 400), 16 * v36, 8);
    }
    v18 = 32LL * *(unsigned int *)(v17 + 384);
    sub_C7D6A0(*(_QWORD *)(v17 + 368), v18, 8);
    v19 = *(_QWORD *)(v17 + 240);
    if ( v19 != v17 + 256 )
      _libc_free(v19, v18);
    v20 = *(_QWORD *)(v17 + 56);
    if ( v20 != v17 + 72 )
      _libc_free(v20, v18);
    v21 = *(unsigned int *)(v17 + 48);
    if ( (_DWORD)v21 )
    {
      v22 = *(_QWORD **)(v17 + 32);
      v23 = &v22[4 * v21];
      do
      {
        if ( *v22 != -4 && *v22 != -16 )
        {
          v24 = v22[1];
          if ( v24 )
            j_j___libc_free_0(v24, v22[3] - v24);
        }
        v22 += 4;
      }
      while ( v23 != v22 );
      v21 = *(unsigned int *)(v17 + 48);
    }
    sub_C7D6A0(*(_QWORD *)(v17 + 32), 32 * v21, 8);
    a2 = 448;
    j_j___libc_free_0(v17, 448);
    v16 = *(_QWORD *)(a1 + 16);
  }
  v25 = sub_22077B0(448);
  if ( v25 )
  {
    *(_BYTE *)v25 = 0;
    a2 = 0x400000000LL;
    *(_QWORD *)(v25 + 168) = v25 + 184;
    *(_QWORD *)(v25 + 8) = v25 + 24;
    *(_QWORD *)(v25 + 296) = v25 + 312;
    *(_QWORD *)(v25 + 16) = 0x200000000LL;
    *(_QWORD *)(v25 + 176) = 0x200000000LL;
    *(_QWORD *)(v25 + 280) = v16;
    *(_QWORD *)(v25 + 288) = a3;
    *(_QWORD *)(v25 + 304) = 0x400000000LL;
    *(_BYTE *)(v25 + 376) = 1;
    *(_QWORD *)(v25 + 384) = v25 + 400;
    *(_QWORD *)(v25 + 392) = 0x200000000LL;
  }
  v26 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v25;
  if ( v26 )
  {
    v27 = *(_QWORD *)(v26 + 384);
    if ( v27 != v26 + 400 )
      _libc_free(v27, a2);
    v28 = *(_QWORD *)(v26 + 296);
    if ( v28 != v26 + 312 )
      _libc_free(v28, a2);
    v29 = *(_QWORD *)(v26 + 168);
    v30 = v29 + 48LL * *(unsigned int *)(v26 + 176);
    if ( v29 != v30 )
    {
      do
      {
        v30 -= 48;
        v31 = *(_QWORD *)(v30 + 16);
        if ( v31 != v30 + 32 )
          _libc_free(v31, a2);
      }
      while ( v29 != v30 );
      v30 = *(_QWORD *)(v26 + 168);
    }
    if ( v30 != v26 + 184 )
      _libc_free(v30, a2);
    v32 = *(_QWORD **)(v26 + 8);
    v33 = &v32[9 * *(unsigned int *)(v26 + 16)];
    if ( v32 != v33 )
    {
      do
      {
        v34 = *(v33 - 7);
        v33 -= 9;
        if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
          sub_BD60C0(v33);
      }
      while ( v32 != v33 );
      v33 = *(_QWORD **)(v26 + 8);
    }
    if ( v33 != (_QWORD *)(v26 + 24) )
      _libc_free(v33, a2);
    j_j___libc_free_0(v26, 448);
  }
  result = sub_D36C10((_QWORD *)a1);
  if ( (_BYTE)result )
  {
    result = sub_D41000(a1, a6, a8, a5, a7);
    *(_BYTE *)(a1 + 40) = result;
  }
  return result;
}
