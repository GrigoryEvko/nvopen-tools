// Function: sub_984D50
// Address: 0x984d50
//
__int64 __fastcall sub_984D50(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  bool v3; // dl
  unsigned int v4; // ecx
  unsigned int v5; // ecx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v11; // [rsp+8h] [rbp-88h]
  __int64 v12; // [rsp+10h] [rbp-80h]
  __int64 v14; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-68h]
  __int64 v16; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-58h]
  __int64 v18; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-48h]
  __int64 v20; // [rsp+50h] [rbp-40h]
  unsigned int v21; // [rsp+58h] [rbp-38h]

  v2 = *(_BYTE *)(a1 - 16);
  v3 = (v2 & 2) != 0;
  if ( (v2 & 2) != 0 )
    v4 = *(_DWORD *)(a1 - 24);
  else
    v4 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v5 = v4 >> 1;
  if ( !v5 )
    return 1;
  v6 = 0;
  v12 = 16LL * (v5 - 1);
  while ( 1 )
  {
    if ( v3 )
      v7 = *(_QWORD *)(a1 - 32);
    else
      v7 = a1 + -16 - 8LL * ((v2 >> 2) & 0xF);
    v9 = *(_QWORD *)(*(_QWORD *)(v7 + v6) + 136LL);
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + v6 + 8) + 136LL);
    v17 = *(_DWORD *)(v8 + 32);
    if ( v17 > 0x40 )
    {
      v11 = v9;
      sub_C43780(&v16, v8 + 24);
      v9 = v11;
    }
    else
    {
      v16 = *(_QWORD *)(v8 + 24);
    }
    v15 = *(_DWORD *)(v9 + 32);
    if ( v15 > 0x40 )
      sub_C43780(&v14, v9 + 24);
    else
      v14 = *(_QWORD *)(v9 + 24);
    sub_AADC30(&v18, &v14, &v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( (unsigned __int8)sub_AB1B10(&v18, a2) )
      break;
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v19 > 0x40 )
    {
      if ( v18 )
        j_j___libc_free_0_0(v18);
    }
    if ( v6 == v12 )
      return 1;
    v2 = *(_BYTE *)(a1 - 16);
    v6 += 16;
    v3 = (v2 & 2) != 0;
  }
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return 0;
}
