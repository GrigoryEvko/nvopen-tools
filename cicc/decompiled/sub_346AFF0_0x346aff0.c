// Function: sub_346AFF0
// Address: 0x346aff0
//
unsigned __int8 *__fastcall sub_346AFF0(__m128i a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int8 *v11; // r14
  unsigned __int16 *v12; // rdx
  unsigned __int64 v13; // r15
  int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rdx
  int v17; // r9d
  __int64 v18; // rdx
  __int64 v19; // rbx
  unsigned int v20; // edx
  __int128 v22; // [rsp-20h] [rbp-140h]
  __int128 v23; // [rsp-10h] [rbp-130h]
  unsigned int v24; // [rsp+8h] [rbp-118h]
  __int64 v25; // [rsp+8h] [rbp-118h]
  __int64 v26; // [rsp+18h] [rbp-108h]
  unsigned int v27; // [rsp+18h] [rbp-108h]
  __int64 v28; // [rsp+20h] [rbp-100h]
  unsigned int v29; // [rsp+28h] [rbp-F8h]
  __int64 v30; // [rsp+40h] [rbp-E0h] BYREF
  int v31; // [rsp+48h] [rbp-D8h]
  unsigned __int16 v32; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v34[2]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE v35[176]; // [rsp+70h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a3 + 80);
  v30 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v30, v8, 1);
  v31 = *(_DWORD *)(a3 + 72);
  v9 = *(_QWORD *)(a3 + 40);
  v10 = *(_QWORD *)(v9 + 40);
  v11 = *(unsigned __int8 **)v9;
  v12 = (unsigned __int16 *)(*(_QWORD *)(v10 + 48) + 16LL * *(unsigned int *)(v9 + 48));
  v13 = *(_QWORD *)(v9 + 8);
  v26 = *(_QWORD *)(v9 + 48);
  v14 = *v12;
  v15 = *(unsigned int *)(a3 + 28);
  v16 = *((_QWORD *)v12 + 1);
  v32 = v14;
  v33 = v16;
  if ( (_WORD)v14 )
  {
    v28 = 0;
    v29 = (unsigned __int16)word_4456580[(unsigned __int16)v14 - 1];
    goto LABEL_5;
  }
  v29 = sub_3009970((__int64)&v32, v15, v16, a5, a6);
  v14 = v32;
  v28 = v18;
  if ( v32 )
  {
LABEL_5:
    if ( (unsigned __int16)(v14 - 176) > 0x34u )
    {
      v17 = word_4456340[v14 - 1];
      goto LABEL_10;
    }
LABEL_18:
    sub_C64ED0("Expanding reductions for scalable vectors is undefined.", 1u);
  }
  if ( sub_3007100((__int64)&v32) )
    goto LABEL_18;
  v17 = sub_3007130((__int64)&v32, v15);
LABEL_10:
  v34[0] = (unsigned __int64)v35;
  v24 = v17;
  v34[1] = 0x800000000LL;
  sub_3408690(a4, v10, v26, (unsigned __int16 *)v34, 0, v17, a1, 0, 0);
  v27 = sub_33CB000(*(_DWORD *)(a3 + 24));
  if ( v24 )
  {
    v19 = 0;
    v25 = 16LL * v24;
    do
    {
      v23 = *(_OWORD *)(v34[0] + v19);
      v19 += 16;
      *((_QWORD *)&v22 + 1) = v13;
      *(_QWORD *)&v22 = v11;
      v11 = sub_3405C90(a4, v27, (__int64)&v30, v29, v28, v15, a1, v22, v23);
      v13 = v20 | v13 & 0xFFFFFFFF00000000LL;
    }
    while ( v25 != v19 );
  }
  if ( (_BYTE *)v34[0] != v35 )
    _libc_free(v34[0]);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v11;
}
