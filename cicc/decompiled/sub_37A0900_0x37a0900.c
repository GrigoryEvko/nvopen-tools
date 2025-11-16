// Function: sub_37A0900
// Address: 0x37a0900
//
_QWORD *__fastcall sub_37A0900(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  unsigned int v8; // r15d
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // r12
  int v14; // r10d
  int v15; // ecx
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // r9
  _QWORD *v18; // r12
  __int64 v20; // rdx
  unsigned __int64 v21; // [rsp+0h] [rbp-70h]
  unsigned __int64 v22; // [rsp+8h] [rbp-68h]
  int v23; // [rsp+18h] [rbp-58h]
  int v24; // [rsp+1Ch] [rbp-54h]
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  int v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)&v25, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v8) = v26;
    v9 = v27;
  }
  else
  {
    v8 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v9 = v20;
  }
  v10 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v12 = *(_QWORD *)(a2 + 80);
  v13 = (_QWORD *)a1[1];
  v14 = *(_DWORD *)(a2 + 100);
  v15 = *(_DWORD *)(a2 + 96);
  v16 = v10;
  v17 = v11;
  v25 = v12;
  if ( v12 )
  {
    v22 = v11;
    v23 = v14;
    v24 = v15;
    v21 = v10;
    sub_B96E90((__int64)&v25, v12, 1);
    v16 = v21;
    v17 = v22;
    v14 = v23;
    v15 = v24;
  }
  v26 = *(_DWORD *)(a2 + 72);
  v18 = sub_33F2D30(v13, (__int64)&v25, v8, v9, v16, v17, v15, v14);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v18;
}
