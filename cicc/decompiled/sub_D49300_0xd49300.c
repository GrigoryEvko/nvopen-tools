// Function: sub_D49300
// Address: 0xd49300
//
__int64 __fastcall sub_D49300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  _BYTE *v7; // r12
  _BYTE *v8; // r14
  __int64 i; // r13
  __int64 result; // rax
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // dl
  _QWORD *v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-58h]
  _BYTE *v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h]
  _BYTE v17[64]; // [rsp+20h] [rbp-40h] BYREF

  v6 = (__int64)&v15;
  v16 = 0x400000000LL;
  v15 = v17;
  sub_D47A20(a1, (__int64)&v15, a3, a4, a5, a6);
  v7 = v15;
  v8 = &v15[8 * (unsigned int)v16];
  if ( v15 == v8 )
    goto LABEL_22;
  for ( i = 0; ; i = result )
  {
    v11 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == *(_QWORD *)v7 + 48LL )
      goto LABEL_26;
    if ( !v11 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_26:
      BUG();
    if ( (*(_BYTE *)(v11 - 17) & 0x20) == 0 || (v6 = 18, (result = sub_B91C10(v11 - 24, 18)) == 0) || i && result != i )
    {
      v8 = v15;
      result = 0;
      goto LABEL_13;
    }
    v7 += 8;
    if ( v8 == v7 )
      break;
  }
  v12 = *(_BYTE *)(result - 16);
  v8 = v15;
  if ( (v12 & 2) != 0 )
  {
    if ( *(_DWORD *)(result - 24) )
    {
      v13 = *(_QWORD **)(result - 32);
      goto LABEL_19;
    }
LABEL_22:
    result = 0;
    goto LABEL_13;
  }
  if ( (*(_WORD *)(result - 16) & 0x3C0) == 0 )
    goto LABEL_22;
  v13 = (_QWORD *)(result - 8LL * ((v12 >> 2) & 0xF) - 16);
LABEL_19:
  if ( *v13 != result )
    result = 0;
LABEL_13:
  if ( v8 != v17 )
  {
    v14 = result;
    _libc_free(v8, v6);
    return v14;
  }
  return result;
}
