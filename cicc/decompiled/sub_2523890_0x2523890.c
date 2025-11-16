// Function: sub_2523890
// Address: 0x2523890
//
__int64 __fastcall sub_2523890(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, __int64 *),
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        _BYTE *a6)
{
  __int64 *v10; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int8 *v14; // rcx
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  unsigned __int8 *v23; // rax
  unsigned __int8 **v26; // [rsp+8h] [rbp-38h]

  v10 = (__int64 *)(a4 + 72);
  v12 = *(_QWORD *)(a4 + 72);
  v13 = v12 & 3;
  v14 = (unsigned __int8 *)(v12 & 0xFFFFFFFFFFFFFFFCLL);
  if ( v13 == 3 )
    v14 = (unsigned __int8 *)*((_QWORD *)v14 + 3);
  v15 = *v14;
  if ( (unsigned __int8)v15 <= 0x1Cu
    || (v20 = (unsigned int)(v15 - 34), (unsigned __int8)v20 > 0x33u)
    || (v21 = 0x8000000000041LL, !_bittest64(&v21, v20)) )
  {
    v16 = sub_25096F0(v10);
    v17 = a3;
    v18 = v16;
    if ( !v16 )
      return 0;
    return sub_25230B0(a1, a2, v17, v18, a5, a4, a6, 0);
  }
  v26 = (unsigned __int8 **)v14;
  v22 = sub_250C680(v10);
  v17 = a3;
  if ( v22 )
  {
    v18 = *(_QWORD *)(v22 + 24);
    if ( !v18 )
      return 0;
    return sub_25230B0(a1, a2, v17, v18, a5, a4, a6, 0);
  }
  v23 = sub_BD3990(*(v26 - 4), 0x8000000000041LL);
  v17 = a3;
  v18 = (__int64)v23;
  if ( !v23 )
    return 0;
  if ( !*v23 )
    return sub_25230B0(a1, a2, v17, v18, a5, a4, a6, 0);
  return 0;
}
