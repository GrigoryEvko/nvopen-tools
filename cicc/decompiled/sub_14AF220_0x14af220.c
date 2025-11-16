// Function: sub_14AF220
// Address: 0x14af220
//
__int64 __fastcall sub_14AF220(__int64 *a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 result; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a1;
  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = v3 + 56;
  if ( (v2 & 4) == 0 )
  {
    if ( (unsigned __int8)sub_1560260(v4, 0, 32) )
      return 1;
    v5 = *(_QWORD *)(v3 - 72);
    if ( *(_BYTE *)(v5 + 16) )
      goto LABEL_8;
    goto LABEL_7;
  }
  if ( (unsigned __int8)sub_1560260(v4, 0, 32) )
    return 1;
  v5 = *(_QWORD *)(v3 - 24);
  if ( !*(_BYTE *)(v5 + 16) )
  {
LABEL_7:
    v9[0] = *(_QWORD *)(v5 + 112);
    if ( !(unsigned __int8)sub_1560260(v9, 0, 32) )
      goto LABEL_8;
    return 1;
  }
LABEL_8:
  result = sub_15603E0((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 56, 0);
  if ( result )
  {
    v7 = (_QWORD *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    v8 = *v7;
    if ( *(_BYTE *)(*v7 + 8LL) == 16 )
      v8 = **(_QWORD **)(v8 + 16);
    return (unsigned int)sub_15E4690(*(_QWORD *)(v7[5] + 56LL), *(_DWORD *)(v8 + 8) >> 8) ^ 1;
  }
  return result;
}
