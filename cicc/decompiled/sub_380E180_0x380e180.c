// Function: sub_380E180
// Address: 0x380e180
//
__int64 __fastcall sub_380E180(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rsi
  _QWORD *v12; // r11
  __int64 v13; // rdx
  unsigned int v14; // esi
  unsigned __int8 *v15; // r13
  unsigned int v16; // r14d
  __int64 v17; // rdx
  __int64 v19; // rdx
  __int128 v20; // [rsp-10h] [rbp-A0h]
  __int64 v21; // [rsp+18h] [rbp-78h]
  _QWORD *v22; // [rsp+18h] [rbp-78h]
  int v23; // [rsp+18h] [rbp-78h]
  _QWORD v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  int v26; // [rsp+38h] [rbp-58h]
  __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  __int64 v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h]
  __int64 v30; // [rsp+58h] [rbp-38h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v27, *a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v8 = v29;
    v9 = (unsigned __int16)v28;
  }
  else
  {
    v9 = v7(*a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v8 = v19;
  }
  v21 = v9;
  v10 = sub_380AAE0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 80);
  v28 = v8;
  v30 = v8;
  v12 = (_QWORD *)a1[1];
  v24[0] = v10;
  v24[1] = v13;
  v27 = v21;
  v29 = v21;
  v25 = v11;
  if ( v11 )
  {
    v22 = v12;
    sub_B96E90((__int64)&v25, v11, 1);
    v12 = v22;
  }
  *((_QWORD *)&v20 + 1) = 1;
  *(_QWORD *)&v20 = v24;
  v14 = *(_DWORD *)(a2 + 24);
  v26 = *(_DWORD *)(a2 + 72);
  v15 = sub_3411BE0(v12, v14, (__int64)&v25, (unsigned __int16 *)&v27, 2, 1, v20);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  v16 = 0;
  v23 = *(_DWORD *)(a2 + 68);
  if ( v23 )
  {
    do
    {
      v17 = v16++;
      v2 = v17 | v2 & 0xFFFFFFFF00000000LL;
      sub_375F650((__int64)a1, a2, v17, (unsigned __int64)v15, v2);
    }
    while ( v23 != v16 );
  }
  return 0;
}
