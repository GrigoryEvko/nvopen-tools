// Function: sub_117C610
// Address: 0x117c610
//
__int64 __fastcall sub_117C610(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  unsigned __int64 v4; // r14
  unsigned __int8 v5; // al
  unsigned __int8 v6; // al
  int v7; // r12d
  __int64 *v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r11
  __int64 v12; // rdi
  __int64 **v13; // rsi
  unsigned int v14; // r12d
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int **v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-80h]
  __int64 **v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  _QWORD v26[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  v3 = *(unsigned __int8 **)(a2 - 64);
  v4 = *(_QWORD *)(a2 - 32);
  v5 = *v3;
  if ( *v3 > 0x15u )
  {
    if ( *(_BYTE *)v4 > 0x15u )
      return 0;
    if ( v5 <= 0x1Cu )
      return 0;
    v7 = v5;
    if ( (unsigned int)v5 - 68 > 1 )
      return 0;
  }
  else
  {
    v6 = *(_BYTE *)v4;
    if ( *(_BYTE *)v4 <= 0x1Cu )
      return 0;
    v7 = v6;
    v3 = *(unsigned __int8 **)(a2 - 32);
    v4 = *(_QWORD *)(a2 - 64);
    if ( (unsigned int)v6 - 68 > 1 )
      return 0;
  }
  if ( (v3[7] & 0x40) != 0 )
    v9 = (__int64 *)*((_QWORD *)v3 - 1);
  else
    v9 = (__int64 *)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  v10 = 0;
  v11 = *(_QWORD *)(*v9 + 8);
  v21 = *v9;
  v12 = v11;
  if ( (unsigned __int8)(**(_BYTE **)(a2 - 96) - 82) <= 1u )
    v10 = *(_QWORD *)(a2 - 96);
  if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    v12 = **(_QWORD **)(v11 + 16);
  v22 = *(__int64 ***)(*v9 + 8);
  v24 = *(_QWORD *)(a2 - 96);
  if ( !sub_BCAC40(v12, 1) && (!v10 || v22 != *(__int64 ***)(*(_QWORD *)(v10 - 64) + 8LL)) )
    return 0;
  v13 = v22;
  v14 = v7 - 29;
  v23 = *(_QWORD *)(a2 + 8);
  v15 = sub_AD4C30(v4, v13, 0);
  v16 = sub_96F480(v14, v15, *(_QWORD *)(v4 + 8), *(_QWORD *)(a1 + 88));
  if ( v16 != v4 || v16 == 0 )
    return 0;
  if ( !v15 )
    return 0;
  v17 = *((_QWORD *)v3 + 2);
  if ( !v17 || *(_QWORD *)(v17 + 8) )
    return 0;
  if ( *(unsigned __int8 **)(a2 - 32) == v3 )
  {
    v20 = v21;
    v21 = v15;
    v15 = v20;
  }
  v18 = *(unsigned int ***)(a1 + 32);
  v27 = 259;
  v26[0] = "narrow";
  v19 = sub_B36550(v18, v24, v21, v15, (__int64)v26, a2);
  v27 = 257;
  return sub_B51D30(v14, v19, v23, (__int64)v26, 0, 0);
}
