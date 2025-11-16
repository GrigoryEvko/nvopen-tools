// Function: sub_11E66A0
// Address: 0x11e66a0
//
__int64 __fastcall sub_11E66A0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _BYTE *v6; // r14
  int v7; // eax
  __int64 v8; // rax
  unsigned __int64 *v9; // rdx
  unsigned int v10; // eax
  unsigned __int64 v11; // rcx
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 *v15; // rax
  __int64 v16; // r8
  __m128i v17; // [rsp+0h] [rbp-80h] BYREF
  __int64 v18; // [rsp+10h] [rbp-70h]
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+20h] [rbp-60h]
  __int64 v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h]
  __int64 v23; // [rsp+38h] [rbp-48h]
  __int16 v24; // [rsp+40h] [rbp-40h]

  v6 = *(_BYTE **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *v6 == 20 )
  {
    v13 = (__int64 *)sub_BD5C60(a2);
    v17.m128i_i32[0] = 0;
    v14 = sub_A77AD0(v13, 0);
    v15 = (__int64 *)sub_BD5C60(a2);
    v16 = v14;
    v6 = 0;
    *(_QWORD *)(a2 + 72) = sub_A7B660((__int64 *)(a2 + 72), v15, &v17, 1, v16);
  }
  else
  {
    v17 = (__m128i)*(unsigned __int64 *)(a1 + 16);
    v18 = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 257;
    if ( !(unsigned __int8)sub_9B6260((__int64)v6, &v17, 0) )
      return 0;
  }
  v7 = *(_DWORD *)(a2 + 4);
  v17 = 0u;
  if ( !(unsigned __int8)sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v7 & 0x7FFFFFF)), &v17, 1u) )
    return 0;
  v8 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v8 != 17 )
    return 0;
  v9 = *(unsigned __int64 **)(v8 + 24);
  v10 = *(_DWORD *)(v8 + 32);
  if ( v10 > 0x40 )
  {
    v11 = *v9;
  }
  else
  {
    v11 = 0;
    if ( v10 )
      v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
  }
  return sub_11DBE40(a2, (char **)&v17, (__int64)v6, v11, a4, a3);
}
