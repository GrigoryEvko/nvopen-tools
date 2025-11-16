// Function: sub_1124890
// Address: 0x1124890
//
unsigned __int8 *__fastcall sub_1124890(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rbx
  unsigned __int8 *v7; // r14
  int v9; // eax
  unsigned __int8 *result; // rax
  __int64 *v11; // rdx
  __int64 v12; // rdx
  _BYTE *v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 *v16; // r14
  __int64 v17; // r13
  __int16 v18; // r12
  __int64 v19; // r14
  __int16 v20; // r12
  unsigned __int8 *v21; // [rsp+8h] [rbp-58h]
  _BYTE v22[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v23; // [rsp+30h] [rbp-30h]

  v6 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v6 > 0x15u )
    return 0;
  v7 = *(unsigned __int8 **)(a2 - 64);
  v9 = *v7;
  if ( (unsigned __int8)v9 <= 0x1Cu )
    return 0;
  if ( v9 != 77 )
  {
    if ( v9 == 84 )
      return sub_F27020((__int64)a1, a2, *(_QWORD *)(a2 - 64), 0, a5, a6);
    if ( v9 == 61 )
    {
      v11 = (v7[7] & 0x40) != 0
          ? (__int64 *)*((_QWORD *)v7 - 1)
          : (__int64 *)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
      v12 = *v11;
      if ( *(_BYTE *)v12 == 63 )
      {
        v13 = *(_BYTE **)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
        if ( *v13 == 3 )
          return (unsigned __int8 *)sub_1123BC0(a1, *(_QWORD *)(a2 - 64), v12, (__int64)v13, a2, 0);
      }
    }
    return 0;
  }
  if ( !sub_AC30F0(*(_QWORD *)(a2 - 32)) )
    return 0;
  v14 = *(_QWORD *)(v6 + 8);
  v15 = sub_AE4450(a1[11], v14);
  v16 = (v7[7] & 0x40) != 0 ? (__int64 *)*((_QWORD *)v7 - 1) : (__int64 *)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
  v17 = *v16;
  if ( v15 != *(_QWORD *)(*v16 + 8) )
    return 0;
  v18 = *(_WORD *)(a2 + 2);
  v19 = sub_AD6530(v15, v14);
  v20 = v18 & 0x3F;
  v23 = 257;
  result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
  if ( result )
  {
    v21 = result;
    sub_1113300((__int64)result, v20, v17, v19, (__int64)v22);
    return v21;
  }
  return result;
}
