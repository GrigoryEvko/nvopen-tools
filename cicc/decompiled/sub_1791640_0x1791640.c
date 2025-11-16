// Function: sub_1791640
// Address: 0x1791640
//
__int64 __fastcall sub_1791640(__int64 a1)
{
  __int64 v1; // rax
  __int64 ***v3; // rcx
  __int64 v4; // r13
  int v5; // r12d
  __int64 v7; // rax
  int v8; // edi
  __int64 ***v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx

  v1 = *(_QWORD *)(a1 - 72);
  if ( *(_BYTE *)(v1 + 16) != 75 )
    return 0;
  v3 = *(__int64 ****)(v1 - 48);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD *)(v1 - 24);
  if ( *(_BYTE *)(v4 + 16) > 0x10u )
    return 0;
  v5 = *(_WORD *)(v1 + 18) & 0x7FFF;
  if ( (unsigned int)(v5 - 32) > 1 )
    return 0;
  v7 = v5 == 32 ? *(_QWORD *)(a1 - 48) : *(_QWORD *)(a1 - 24);
  v8 = *(unsigned __int8 *)(v7 + 16);
  if ( (unsigned __int8)v8 <= 0x17u || (unsigned int)(v8 - 35) > 0x11 )
    return 0;
  v9 = *(__int64 ****)(v7 - 48);
  v10 = *(_QWORD *)(v7 - 24);
  if ( v3 == v9 )
  {
    if ( !v10 )
      return 0;
  }
  else
  {
    if ( !v9 || (__int64 ***)v10 != v3 )
      return 0;
    v10 = *(_QWORD *)(v7 - 48);
  }
  if ( v4 != sub_15A14F0(v8 - 24, *v3, 0) )
    return 0;
  v11 = -48;
  if ( v5 != 32 )
    v11 = -24;
  v12 = (_QWORD *)(a1 + v11);
  if ( *v12 )
  {
    v13 = v12[1];
    v14 = v12[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v14 = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
  }
  *v12 = v10;
  v15 = *(_QWORD *)(v10 + 8);
  v12[1] = v15;
  if ( v15 )
    *(_QWORD *)(v15 + 16) = (unsigned __int64)(v12 + 1) | *(_QWORD *)(v15 + 16) & 3LL;
  v12[2] = (v10 + 8) | v12[2] & 3LL;
  *(_QWORD *)(v10 + 8) = v12;
  return a1;
}
