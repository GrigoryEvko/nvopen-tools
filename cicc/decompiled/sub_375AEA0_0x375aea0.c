// Function: sub_375AEA0
// Address: 0x375aea0
//
void __fastcall sub_375AEA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v13; // rax
  unsigned __int16 v14; // si
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int128 v19; // [rsp+0h] [rbp-80h] BYREF
  __int64 v20; // [rsp+10h] [rbp-70h] BYREF
  int v21; // [rsp+18h] [rbp-68h]
  unsigned int v22; // [rsp+20h] [rbp-60h] BYREF
  __int64 v23; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v24; // [rsp+30h] [rbp-50h] BYREF
  int v25; // [rsp+38h] [rbp-48h]
  __int64 v26; // [rsp+40h] [rbp-40h]
  int v27; // [rsp+48h] [rbp-38h]

  *(_QWORD *)&v19 = a2;
  v9 = *(_QWORD *)(a2 + 80);
  *((_QWORD *)&v19 + 1) = a3;
  v20 = v9;
  if ( v9 )
  {
    sub_B96E90((__int64)&v20, v9, 1);
    v10 = v19;
  }
  else
  {
    v10 = a2;
  }
  v11 = *a1;
  v21 = *(_DWORD *)(a2 + 72);
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v11 + 592LL);
  v13 = *(_QWORD *)(v10 + 48) + 16LL * DWORD2(v19);
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v16 = a1[1];
  if ( v12 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v24, v11, *(_QWORD *)(v16 + 64), v14, v15);
    LOWORD(v22) = v25;
    v23 = v26;
  }
  else
  {
    v22 = v12(v11, *(_QWORD *)(v16 + 64), v14, v15);
    v23 = v18;
  }
  sub_34081D0(&v24, (_QWORD *)a1[1], &v19, (__int64)&v20, &v22, &v22, a6);
  v17 = v20;
  *(_QWORD *)a4 = v24;
  *(_DWORD *)(a4 + 8) = v25;
  *(_QWORD *)a5 = v26;
  *(_DWORD *)(a5 + 8) = v27;
  if ( v17 )
    sub_B91220((__int64)&v20, v17);
}
