// Function: sub_3815CB0
// Address: 0x3815cb0
//
unsigned __int8 *__fastcall sub_3815CB0(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  unsigned int v8; // r13d
  __int64 v9; // r15
  __int64 v10; // rsi
  _QWORD *v11; // r9
  __int128 *v12; // r12
  unsigned int v13; // esi
  unsigned __int8 *v14; // r12
  __int64 v16; // rdx
  _QWORD *v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  int v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)&v18, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v8) = v19;
    v9 = v20;
  }
  else
  {
    v8 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v9 = v16;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = (_QWORD *)a1[1];
  v12 = *(__int128 **)(a2 + 40);
  v18 = v10;
  if ( v10 )
  {
    v17 = v11;
    sub_B96E90((__int64)&v18, v10, 1);
    v11 = v17;
  }
  v13 = *(_DWORD *)(a2 + 24);
  v19 = *(_DWORD *)(a2 + 72);
  v14 = sub_3406EB0(v11, v13, (__int64)&v18, v8, v9, (__int64)v11, *v12, *(__int128 *)((char *)v12 + 40));
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v14;
}
