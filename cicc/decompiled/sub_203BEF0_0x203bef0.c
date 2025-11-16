// Function: sub_203BEF0
// Address: 0x203bef0
//
_QWORD *__fastcall sub_203BEF0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v6; // r14
  char *v7; // rax
  __int64 v8; // rsi
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // rsi
  char v12; // r12
  int v13; // ebx
  int v14; // r8d
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // r9d
  _QWORD *v18; // r10
  int v19; // r8d
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // r13
  unsigned int v23; // esi
  int v24; // r11d
  int v25; // r12d
  int v26; // r13d
  const void *v27; // r15
  _QWORD *v28; // r12
  int v30; // [rsp+8h] [rbp-F8h]
  int v31; // [rsp+Ch] [rbp-F4h]
  _QWORD *v32; // [rsp+10h] [rbp-F0h]
  int v33; // [rsp+18h] [rbp-E8h]
  __int64 v34; // [rsp+20h] [rbp-E0h]
  __int64 v35; // [rsp+28h] [rbp-D8h]
  __int64 v36; // [rsp+30h] [rbp-D0h]
  __int64 v37; // [rsp+38h] [rbp-C8h]
  _DWORD *v38; // [rsp+48h] [rbp-B8h]
  __int64 v39; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+58h] [rbp-A8h]
  __int64 v41; // [rsp+60h] [rbp-A0h] BYREF
  int v42; // [rsp+68h] [rbp-98h]
  unsigned int v43; // [rsp+70h] [rbp-90h] BYREF
  const void **v44; // [rsp+78h] [rbp-88h]
  _DWORD *v45; // [rsp+80h] [rbp-80h] BYREF
  __int64 v46; // [rsp+88h] [rbp-78h]
  _QWORD v47[14]; // [rsp+90h] [rbp-70h] BYREF

  v6 = a1;
  v7 = *(char **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v41 = v8;
  v40 = v10;
  LOBYTE(v39) = v9;
  if ( v8 )
    sub_1623A60((__int64)&v41, v8, 2);
  v11 = *a1;
  v42 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v45, v11, *(_QWORD *)(a1[1] + 48), v39, v40);
  v12 = v46;
  v44 = (const void **)v47[0];
  LOBYTE(v43) = v46;
  if ( (_BYTE)v39 )
    v13 = word_4305480[(unsigned __int8)(v39 - 14)];
  else
    v13 = sub_1F58D30((__int64)&v39);
  if ( v12 )
    v14 = word_4305480[(unsigned __int8)(v12 - 14)];
  else
    v14 = sub_1F58D30((__int64)&v43);
  v33 = v14;
  v36 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v37 = v15;
  v16 = sub_20363F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v18 = v47;
  v19 = v33;
  v34 = v16;
  v35 = v20;
  v21 = 0;
  v45 = v47;
  v46 = 0x1000000000LL;
  if ( v13 )
  {
    v22 = 0;
    v23 = 16;
    v21 = 0;
    v24 = v33 - v13;
    while ( 1 )
    {
      v25 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + v22);
      if ( v13 <= v25 )
        v25 += v24;
      if ( v23 <= (unsigned int)v21 )
      {
        v30 = v24;
        v31 = v19;
        v32 = v18;
        sub_16CD150((__int64)&v45, v18, 0, 4, v19, v17);
        v21 = (unsigned int)v46;
        v24 = v30;
        v19 = v31;
        v18 = v32;
      }
      v45[v21] = v25;
      v21 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
      if ( 4LL * (unsigned int)(v13 - 1) == v22 )
        break;
      v23 = HIDWORD(v46);
      v22 += 4;
    }
    v6 = a1;
  }
  if ( v19 != v13 )
  {
    v26 = v19;
    v27 = v18;
    do
    {
      if ( HIDWORD(v46) <= (unsigned int)v21 )
      {
        sub_16CD150((__int64)&v45, v27, 0, 4, v19, v17);
        v21 = (unsigned int)v46;
      }
      ++v13;
      v45[v21] = -1;
      v21 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
    }
    while ( v13 != v26 );
    v18 = v27;
  }
  v38 = v18;
  v28 = sub_1D41320(v6[1], v43, v44, (__int64)&v41, v36, v37, a3, a4, a5, v34, v35, v45, (unsigned int)v21);
  if ( v45 != v38 )
    _libc_free((unsigned __int64)v45);
  if ( v41 )
    sub_161E7C0((__int64)&v41, v41);
  return v28;
}
