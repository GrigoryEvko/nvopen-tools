// Function: sub_3871090
// Address: 0x3871090
//
void __fastcall sub_3871090(__int64 *a1, __int64 **a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v7; // r14
  __int16 i; // ax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 *v13; // r14
  _BYTE *v14; // rsi
  __int64 v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-94h]
  __int64 v17; // [rsp+10h] [rbp-90h]
  __int64 v18; // [rsp+18h] [rbp-88h]
  _QWORD *v19; // [rsp+20h] [rbp-80h] BYREF
  __int64 v20; // [rsp+28h] [rbp-78h]
  _QWORD v21[14]; // [rsp+30h] [rbp-70h] BYREF

  v7 = *a1;
  for ( i = *(_WORD *)(*a1 + 24); i == 7; i = *(_WORD *)(*a1 + 24) )
  {
    *a1 = **(_QWORD **)(v7 + 32);
    v16 = *(_WORD *)(v7 + 26) & 1;
    v17 = *(_QWORD *)(v7 + 48);
    v18 = sub_13A5BC0((_QWORD *)v7, a3);
    v9 = sub_1456040(**(_QWORD **)(v7 + 32));
    v10 = sub_145CF80(a3, v9, 0, 0);
    v11 = sub_14799E0(a3, v10, v18, v17, v16);
    v12 = *a2;
    v21[1] = v11;
    v21[0] = v12;
    v19 = v21;
    v20 = 0x200000002LL;
    v13 = sub_147DD40(a3, (__int64 *)&v19, 0, 0, a4, a5);
    if ( v19 != v21 )
      _libc_free((unsigned __int64)v19);
    *a2 = v13;
    v7 = *a1;
  }
  if ( i == 4 )
  {
    *a1 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 8LL * ((unsigned int)*(_QWORD *)(v7 + 40) - 1));
    v14 = *(_BYTE **)(v7 + 32);
    v15 = *(_QWORD *)(v7 + 40);
    v19 = v21;
    v20 = 0x800000000LL;
    sub_145C5B0((__int64)&v19, v14, &v14[8 * v15]);
    v19[(unsigned int)v20 - 1] = *a2;
    *a2 = sub_147DD40(a3, (__int64 *)&v19, 0, 0, a4, a5);
    sub_3871090(a1, a2, a3);
    if ( v19 != v21 )
      _libc_free((unsigned __int64)v19);
  }
}
