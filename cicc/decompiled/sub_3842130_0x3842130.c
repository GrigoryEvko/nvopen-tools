// Function: sub_3842130
// Address: 0x3842130
//
unsigned __int8 *__fastcall sub_3842130(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v8; // r13d
  __int64 v9; // rsi
  __int64 v10; // rsi
  unsigned __int16 *v11; // rax
  int v12; // r9d
  int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rsi
  unsigned __int8 *v16; // r12
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+40h] [rbp-60h] BYREF
  int v22; // [rsp+48h] [rbp-58h]
  _BYTE v23[8]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+60h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)v23, *a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    LOWORD(v8) = v24;
    v20 = v25;
  }
  else
  {
    v18 = v7(*a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v20 = v19;
    v8 = v18;
  }
  v9 = *(_QWORD *)(a2 + 80);
  v21 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v21, v9, 1);
  v10 = *a1;
  v22 = *(_DWORD *)(a2 + 72);
  v11 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  sub_2FE6CC0((__int64)v23, v10, *(_QWORD *)(a1[1] + 64), *v11, *((_QWORD *)v11 + 1));
  if ( v23[0] == 1 )
  {
    v13 = *(_DWORD *)(a2 + 24);
    switch ( v13 )
    {
      case 224:
        sub_383B380((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
        break;
      case 225:
        sub_37AF270((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
        break;
      case 223:
        sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
        break;
      default:
        BUG();
    }
    v14 = a1[1];
    v15 = *(unsigned int *)(a2 + 24);
  }
  else
  {
    v14 = a1[1];
    v15 = *(unsigned int *)(a2 + 24);
  }
  v16 = sub_33FAF80(v14, v15, (__int64)&v21, v8, v20, v12, a3);
  if ( v21 )
    sub_B91220((__int64)&v21, v21);
  return v16;
}
