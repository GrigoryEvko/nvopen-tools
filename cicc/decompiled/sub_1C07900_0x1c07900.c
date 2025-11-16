// Function: sub_1C07900
// Address: 0x1c07900
//
__int64 __fastcall sub_1C07900(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rax
  char v9; // dl
  unsigned int v10; // r14d
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = sub_1560260((_QWORD *)(a1 + 56), -1, 36);
  if ( (_BYTE)v1 )
    return 1;
  v2 = v1;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_21;
  v4 = sub_1648A40(a1);
  v6 = v4 + v5;
  v7 = 0;
  if ( *(char *)(a1 + 23) < 0 )
    v7 = sub_1648A40(a1);
  if ( !(unsigned int)((v6 - v7) >> 4) )
  {
LABEL_21:
    v8 = *(_QWORD *)(a1 - 24);
    v9 = *(_BYTE *)(v8 + 16);
    if ( v9 )
      goto LABEL_17;
    v11[0] = *(_QWORD *)(v8 + 112);
    if ( (unsigned __int8)sub_1560260(v11, -1, 36) )
      return 1;
  }
  v8 = *(_QWORD *)(a1 - 24);
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 )
    goto LABEL_17;
  if ( (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
    return v2;
  v10 = *(_DWORD *)(v8 + 36);
  if ( (unsigned __int8)sub_1C30240(v10) || v10 == 149 || v10 == 215 || v10 == 3 || (unsigned __int8)sub_1C301F0(v10) )
    return 1;
  v8 = *(_QWORD *)(a1 - 24);
  v9 = *(_BYTE *)(v8 + 16);
LABEL_17:
  if ( v9 != 20 )
    return v2;
  return *(unsigned __int8 *)(v8 + 96) ^ 1u;
}
