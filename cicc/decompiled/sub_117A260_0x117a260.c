// Function: sub_117A260
// Address: 0x117a260
//
_QWORD *__fastcall sub_117A260(__int64 a1, unsigned int **a2)
{
  _BYTE *v2; // r12
  unsigned __int8 *v3; // r15
  unsigned __int8 *v4; // r13
  unsigned __int8 *v5; // r14
  int v7; // eax
  _QWORD *result; // rax
  int v9; // eax
  unsigned __int8 *v10; // rsi
  __int64 v11; // rsi
  int v12; // eax
  int v13; // eax
  unsigned __int8 *v14; // rdi
  __int64 v15; // rdi
  int v16; // eax
  int v17; // eax
  unsigned __int8 *v18; // r15
  __int64 v19; // r8
  int v20; // eax
  int v21; // eax
  unsigned __int8 *v22; // rcx
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // r13
  unsigned __int8 *v27; // [rsp+8h] [rbp-68h]
  _QWORD *v28; // [rsp+8h] [rbp-68h]
  _BYTE v29[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v30; // [rsp+30h] [rbp-40h]

  v2 = *(_BYTE **)(a1 - 96);
  v3 = *(unsigned __int8 **)(a1 - 64);
  if ( (unsigned __int8)(*v2 - 82) > 1u )
    return 0;
  v4 = (unsigned __int8 *)*((_QWORD *)v2 - 8);
  v27 = *(unsigned __int8 **)(a1 - 32);
  if ( !v4 )
    return 0;
  v5 = (unsigned __int8 *)*((_QWORD *)v2 - 4);
  if ( !v5 )
    return 0;
  sub_B53900((__int64)v2);
  if ( v3 == v4 || v3 == v5 || v27 == v4 || v27 == v5 )
    return 0;
  v7 = *v4;
  if ( (unsigned __int8)v7 > 0x1Cu )
  {
    v9 = v7 - 29;
  }
  else
  {
    if ( (_BYTE)v7 != 5 )
      return 0;
    v9 = *((unsigned __int16 *)v4 + 1);
  }
  if ( v9 != 49 )
    return 0;
  v10 = (v4[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v4 - 1) : &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v11 = *(_QWORD *)v10;
  if ( !v11 )
    return 0;
  v12 = *v5;
  if ( (unsigned __int8)v12 > 0x1Cu )
  {
    v13 = v12 - 29;
  }
  else
  {
    if ( (_BYTE)v12 != 5 )
      return 0;
    v13 = *((unsigned __int16 *)v5 + 1);
  }
  if ( v13 != 49 )
    return 0;
  v14 = (v5[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v5 - 1) : &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
  v15 = *(_QWORD *)v14;
  if ( !v15 )
    return 0;
  v16 = *v3;
  if ( (unsigned __int8)v16 > 0x1Cu )
  {
    v17 = v16 - 29;
  }
  else
  {
    if ( (_BYTE)v16 != 5 )
      return 0;
    v17 = *((unsigned __int16 *)v3 + 1);
  }
  if ( v17 != 49 )
    return 0;
  v18 = (v3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v3 - 1) : &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  v19 = *(_QWORD *)v18;
  if ( !*(_QWORD *)v18 )
    return 0;
  v20 = *v27;
  if ( (unsigned __int8)v20 > 0x1Cu )
  {
    v21 = v20 - 29;
  }
  else
  {
    if ( (_BYTE)v20 != 5 )
      return 0;
    v21 = *((unsigned __int16 *)v27 + 1);
  }
  if ( v21 != 49 )
    return 0;
  v22 = (v27[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v27 - 1) : &v27[-32 * (*((_DWORD *)v27 + 1) & 0x7FFFFFF)];
  v23 = *(_QWORD *)v22;
  if ( !*(_QWORD *)v22 )
    return 0;
  if ( v11 == v19 && v15 == v23 )
  {
    v30 = 257;
    v24 = sub_B36550(a2, (__int64)v2, (__int64)v4, (__int64)v5, (__int64)v29, a1);
  }
  else
  {
    if ( v15 != v19 || v11 != v23 )
      return 0;
    v30 = 257;
    v24 = sub_B36550(a2, (__int64)v2, (__int64)v5, (__int64)v4, (__int64)v29, a1);
  }
  v25 = *(_QWORD *)(a1 + 8);
  v30 = 257;
  result = sub_BD2C40(72, unk_3F10A14);
  if ( result )
  {
    v28 = result;
    sub_B51BF0((__int64)result, v24, v25, (__int64)v29, 0, 0);
    return v28;
  }
  return result;
}
