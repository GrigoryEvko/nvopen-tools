// Function: sub_38155A0
// Address: 0x38155a0
//
unsigned __int8 *__fastcall sub_38155A0(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // rsi
  int v13; // eax
  _QWORD *v14; // rdi
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned __int8 *v17; // r14
  __int64 v19; // rdx
  __int128 v20; // [rsp-10h] [rbp-80h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h] BYREF
  int v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]
  __int64 v27; // [rsp+38h] [rbp-38h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v24, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v10 = (unsigned __int16)v25;
    v11 = v26;
  }
  else
  {
    v10 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v11 = v19;
  }
  v12 = *(_QWORD *)(a2 + 80);
  v22 = v12;
  if ( v12 )
  {
    v21 = v11;
    sub_B96E90((__int64)&v22, v12, 1);
    v11 = v21;
  }
  v13 = *(_DWORD *)(a2 + 72);
  v25 = v11;
  v14 = (_QWORD *)a1[1];
  v15 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v20 + 1) = 1;
  v23 = v13;
  v16 = *(_QWORD *)(a2 + 40);
  LOWORD(v26) = 1;
  *(_QWORD *)&v20 = v16;
  v24 = v10;
  v27 = 0;
  v17 = sub_3411BE0(v14, v15, (__int64)&v22, (unsigned __int16 *)&v24, 2, v9, v20);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v17, 1);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v17;
}
