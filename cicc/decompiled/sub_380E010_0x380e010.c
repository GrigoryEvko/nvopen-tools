// Function: sub_380E010
// Address: 0x380e010
//
unsigned __int8 *__fastcall sub_380E010(__int64 *a1, unsigned __int64 a2)
{
  __int16 *v3; // rax
  unsigned __int16 v4; // si
  __int64 v5; // r8
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  _QWORD *v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // esi
  unsigned __int8 *v16; // r14
  __int64 v18; // rdx
  __int128 v19; // [rsp-10h] [rbp-90h]
  __int64 v20; // [rsp+0h] [rbp-80h]
  _QWORD v21[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  int v23; // [rsp+28h] [rbp-58h]
  __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+38h] [rbp-48h]
  __int64 v26; // [rsp+40h] [rbp-40h]
  __int64 v27; // [rsp+48h] [rbp-38h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v6 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v24, *a1, *(_QWORD *)(a1[1] + 64), v4, v5);
    v7 = (unsigned __int16)v25;
    v8 = v26;
  }
  else
  {
    v7 = v6(*a1, *(_QWORD *)(a1[1] + 64), v4, v5);
    v8 = v18;
  }
  v20 = v8;
  v9 = sub_380AAE0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v24 = v7;
  v21[0] = v9;
  v10 = *(_QWORD *)(a2 + 48);
  v11 = *(_QWORD *)(a2 + 80);
  v12 = (_QWORD *)a1[1];
  v25 = v20;
  v21[1] = v13;
  LOWORD(v13) = *(_WORD *)(v10 + 16);
  v14 = *(_QWORD *)(v10 + 24);
  v22 = v11;
  LOWORD(v26) = v13;
  v27 = v14;
  if ( v11 )
    sub_B96E90((__int64)&v22, v11, 1);
  *((_QWORD *)&v19 + 1) = 1;
  *(_QWORD *)&v19 = v21;
  v15 = *(_DWORD *)(a2 + 24);
  v23 = *(_DWORD *)(a2 + 72);
  v16 = sub_3411BE0(v12, v15, (__int64)&v22, (unsigned __int16 *)&v24, 2, 1, v19);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v16, 1);
  return v16;
}
