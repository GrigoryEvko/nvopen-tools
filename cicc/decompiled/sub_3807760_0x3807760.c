// Function: sub_3807760
// Address: 0x3807760
//
unsigned __int8 *__fastcall sub_3807760(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  char v12; // al
  unsigned int v13; // eax
  unsigned int v14; // ebx
  __int64 v15; // rax
  _QWORD *v16; // rbx
  __int64 v17; // rdx
  __int128 v18; // rax
  unsigned __int8 *v19; // r12
  __int64 v21; // rax
  __int64 v22; // rdx
  __int128 v23; // [rsp+0h] [rbp-A0h]
  unsigned int v24; // [rsp+10h] [rbp-90h] BYREF
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+20h] [rbp-80h] BYREF
  int v27; // [rsp+28h] [rbp-78h]
  unsigned __int64 v28; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-68h]
  __int64 v30; // [rsp+40h] [rbp-60h] BYREF
  char v31; // [rsp+48h] [rbp-58h]
  __int64 v32; // [rsp+50h] [rbp-50h] BYREF
  __int64 v33; // [rsp+58h] [rbp-48h]
  __int64 v34; // [rsp+60h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v32, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v24) = v33;
    v25 = v34;
  }
  else
  {
    v24 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v25 = v22;
  }
  v9 = *(_QWORD *)(a2 + 80);
  v26 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v26, v9, 1);
  v27 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v24 )
  {
    if ( (_WORD)v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
      BUG();
    v21 = 16LL * ((unsigned __int16)v24 - 1);
    v11 = *(_QWORD *)&byte_444C4A0[v21];
    v12 = byte_444C4A0[v21 + 8];
  }
  else
  {
    v32 = sub_3007260((__int64)&v24);
    v33 = v10;
    v11 = v32;
    v12 = v33;
  }
  v30 = v11;
  v31 = v12;
  v13 = sub_CA1930(&v30);
  v29 = v13;
  v14 = v13;
  if ( v13 > 0x40 )
  {
    sub_C43690((__int64)&v28, 0, 0);
    v15 = 1LL << ((unsigned __int8)v14 - 1);
    if ( v29 > 0x40 )
    {
      *(_QWORD *)(v28 + 8LL * ((v14 - 1) >> 6)) |= v15;
      goto LABEL_10;
    }
  }
  else
  {
    v28 = 0;
    v15 = 1LL << ((unsigned __int8)v13 - 1);
  }
  v28 |= v15;
LABEL_10:
  v16 = (_QWORD *)a1[1];
  *(_QWORD *)&v23 = sub_34007B0((__int64)v16, (__int64)&v28, (__int64)&v26, v24, v25, 0, a3, 0);
  *((_QWORD *)&v23 + 1) = v17;
  *(_QWORD *)&v18 = sub_3805E70((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v19 = sub_3406EB0(v16, 0xBCu, (__int64)&v26, v24, v25, *((__int64 *)&v23 + 1), v18, v23);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return v19;
}
