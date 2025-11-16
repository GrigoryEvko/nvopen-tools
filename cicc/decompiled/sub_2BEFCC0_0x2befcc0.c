// Function: sub_2BEFCC0
// Address: 0x2befcc0
//
void *__fastcall sub_2BEFCC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 *v12; // rdi
  __int64 *v13; // rdi
  __int64 v15[2]; // [rsp+0h] [rbp-180h] BYREF
  __int64 v16; // [rsp+10h] [rbp-170h] BYREF
  __int64 v17[2]; // [rsp+20h] [rbp-160h] BYREF
  __int64 v18; // [rsp+30h] [rbp-150h] BYREF
  __int64 *v19; // [rsp+40h] [rbp-140h] BYREF
  __int16 v20; // [rsp+60h] [rbp-120h]
  __m128i v21[3]; // [rsp+70h] [rbp-110h] BYREF
  __m128i v22[3]; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v23[2]; // [rsp+D0h] [rbp-B0h] BYREF
  char v24; // [rsp+F0h] [rbp-90h]
  char v25; // [rsp+F1h] [rbp-8Fh]
  __m128i v26; // [rsp+100h] [rbp-80h] BYREF
  __int16 v27; // [rsp+120h] [rbp-60h]
  __m128i v28[2]; // [rsp+130h] [rbp-50h] BYREF
  char v29; // [rsp+150h] [rbp-30h]
  char v30; // [rsp+151h] [rbp-2Fh]

  v30 = 1;
  v28[0].m128i_i64[0] = (__int64)">";
  v29 = 3;
  v26.m128i_i64[0] = (__int64)sub_BD5D20(a2);
  v26.m128i_i64[1] = v4;
  v27 = 261;
  v23[0].m128i_i64[0] = (__int64)"ir-bb<";
  v25 = 1;
  v24 = 3;
  sub_9C6370(v22, v23, &v26, 261, v5, v6);
  sub_9C6370(v21, v22, v28, v7, v8, v9);
  sub_CA0F50(v15, (void **)v21);
  v19 = v15;
  v20 = 260;
  sub_CA0F50(v17, (void **)&v19);
  v10 = (_BYTE *)v17[0];
  *(_BYTE *)(a1 + 8) = 2;
  v11 = v17[1];
  *(_QWORD *)a1 = &unk_4A23970;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  sub_2BEF590((__int64 *)(a1 + 16), v10, (__int64)&v10[v11]);
  v12 = (__int64 *)v17[0];
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x100000000LL;
  *(_QWORD *)(a1 + 88) = 0x100000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 104) = 0;
  if ( v12 != &v18 )
    j_j___libc_free_0((unsigned __int64)v12);
  v13 = (__int64 *)v15[0];
  *(_QWORD *)a1 = &unk_4A23A00;
  *(_QWORD *)(a1 + 120) = a1 + 112;
  *(_QWORD *)(a1 + 112) = (a1 + 112) | 4;
  if ( v13 != &v16 )
    j_j___libc_free_0((unsigned __int64)v13);
  *(_QWORD *)(a1 + 128) = a2;
  *(_QWORD *)a1 = &unk_4A239C8;
  return &unk_4A239C8;
}
