// Function: sub_379C210
// Address: 0x379c210
//
__int64 __fastcall sub_379C210(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r9
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // r13
  __int64 v12; // rdx
  __int128 v13; // rax
  __int64 v14; // r12
  __int64 v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-70h]
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  int v20; // [rsp+18h] [rbp-58h]
  _BYTE v21[8]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v19 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v19, v3, 1);
  v4 = *a1;
  v20 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    HIWORD(v10) = 0;
    sub_2FE6CC0((__int64)v21, v4, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v10) = v22;
    v11 = v23;
  }
  else
  {
    v10 = v5(v4, *(_QWORD *)(v9 + 64), v7, v8);
    v11 = v16;
  }
  v17 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v18 = v12;
  *(_QWORD *)&v13 = sub_379AB60(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v14 = sub_340EC60(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v19,
          v10,
          v11,
          *(unsigned int *)(a2 + 28),
          v17,
          v18,
          v13,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v14;
}
