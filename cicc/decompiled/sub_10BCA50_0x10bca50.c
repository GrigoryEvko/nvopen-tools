// Function: sub_10BCA50
// Address: 0x10bca50
//
__int64 __fastcall sub_10BCA50(__int64 a1, __int64 a2, char a3, char a4, unsigned int **a5, __int64 *a6)
{
  _BYTE *v6; // r14
  unsigned __int8 *v8; // r15
  _BYTE *v10; // r11
  _BYTE *v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // r14d
  _BYTE *v16; // r11
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v24; // [rsp+10h] [rbp-90h]
  _BYTE *v25; // [rsp+10h] [rbp-90h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  _BYTE *v27; // [rsp+10h] [rbp-90h]
  int v30; // [rsp+38h] [rbp-68h]
  _BYTE v31[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v32; // [rsp+60h] [rbp-40h]

  if ( !a1 )
    return 0;
  v6 = *(_BYTE **)(a1 - 64);
  if ( !v6 )
    return 0;
  v8 = *(unsigned __int8 **)(a1 - 32);
  if ( *v8 > 0x15u )
    return 0;
  v24 = sub_B53900(a1);
  if ( !sub_98ED60(v8, 0, 0, 0, 0) || *v6 <= 0x15u )
    return 0;
  if ( a3 )
  {
    if ( v24 != 32 )
      return 0;
  }
  else if ( v24 != 33 )
  {
    return 0;
  }
  if ( !a2 )
    return 0;
  v10 = *(_BYTE **)(a2 - 64);
  v11 = *(_BYTE **)(a2 - 32);
  if ( !v10 || !v11 )
    return 0;
  if ( v11 == v6 )
  {
    v27 = *(_BYTE **)(a2 - 64);
    v22 = sub_B53900(a2);
    v16 = v27;
    v14 = v22 & 0xFFFFFFFFFFLL;
    v15 = v22;
  }
  else
  {
    if ( v6 != v10 )
      return 0;
    v25 = *(_BYTE **)(a2 - 32);
    v13 = sub_B53960(a2);
    v14 = v13 & 0xFFFFFFFFFFLL;
    v15 = v13;
    v16 = v25;
  }
  v26 = (__int64)v16;
  v17 = sub_1016CC0(v15 | v14 & 0xFFFFFFFF00000000LL, v16, v8, a6);
  if ( !v17 )
  {
    v20 = *(_QWORD *)(a2 + 16);
    if ( !v20 || *(_QWORD *)(v20 + 8) )
      return 0;
    v32 = 257;
    v17 = sub_92B530(a5, v15, v26, v8, (__int64)v31);
  }
  if ( a4 )
  {
    v32 = 257;
    v18 = *(_QWORD *)(v17 + 8);
    if ( a3 )
    {
      v19 = sub_AD6530(v18, 257);
      return sub_B36550(a5, a1, v17, v19, (__int64)v31, 0);
    }
    else
    {
      v21 = sub_AD62B0(v18);
      return sub_B36550(a5, a1, v21, v17, (__int64)v31, 0);
    }
  }
  else
  {
    v32 = 257;
    return sub_10BBE20((__int64 *)a5, (unsigned int)(a3 == 0) + 28, a1, v17, v30, 0, (__int64)v31, 0);
  }
}
