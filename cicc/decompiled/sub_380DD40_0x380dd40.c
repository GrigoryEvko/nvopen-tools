// Function: sub_380DD40
// Address: 0x380dd40
//
__int64 __fastcall sub_380DD40(__int64 *a1, __int64 a2)
{
  __int16 *v3; // rax
  unsigned __int16 v4; // si
  __int64 v5; // r8
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v7; // r8
  unsigned int v8; // ecx
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r15
  __int128 v13; // rax
  __int64 v14; // r9
  _QWORD *v15; // r10
  unsigned int v16; // ecx
  __int64 v17; // r8
  unsigned int v18; // esi
  __int64 v19; // r12
  __int64 v21; // rdx
  __int128 v22; // [rsp-20h] [rbp-B0h]
  __int64 v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  unsigned int v25; // [rsp+10h] [rbp-80h]
  unsigned int v26; // [rsp+18h] [rbp-78h]
  _QWORD *v27; // [rsp+18h] [rbp-78h]
  __int128 v28; // [rsp+20h] [rbp-70h]
  __int128 v29; // [rsp+30h] [rbp-60h]
  __int64 v30; // [rsp+40h] [rbp-50h] BYREF
  int v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v6 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v30, *a1, *(_QWORD *)(a1[1] + 64), v4, v5);
    v7 = v32;
    v8 = (unsigned __int16)v31;
  }
  else
  {
    v8 = v6(*a1, *(_QWORD *)(a1[1] + 64), v4, v5);
    v7 = v21;
  }
  v24 = v7;
  v26 = v8;
  *(_QWORD *)&v29 = sub_380AAE0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v29 + 1) = v9;
  v10 = sub_380AAE0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v12 = v11;
  *(_QWORD *)&v13 = sub_380AAE0(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v15 = (_QWORD *)a1[1];
  v16 = v26;
  v17 = v24;
  v28 = v13;
  v30 = *(_QWORD *)(a2 + 80);
  if ( v30 )
  {
    v23 = v24;
    v25 = v26;
    v27 = v15;
    sub_B96E90((__int64)&v30, v30, 1);
    v17 = v23;
    v16 = v25;
    v15 = v27;
  }
  v18 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v22 + 1) = v12;
  *(_QWORD *)&v22 = v10;
  v31 = *(_DWORD *)(a2 + 72);
  v19 = sub_340F900(v15, v18, (__int64)&v30, v16, v17, v14, v29, v22, v28);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v19;
}
