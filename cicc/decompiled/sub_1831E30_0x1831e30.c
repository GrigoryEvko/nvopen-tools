// Function: sub_1831E30
// Address: 0x1831e30
//
__int64 __fastcall sub_1831E30(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // r13
  __int64 result; // rax
  _QWORD *v6; // rdi
  char v7; // r8
  __int64 v8; // rax
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = (__int64 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  if ( ((a2 >> 2) & 1) != 0 )
    v3 = (__int64 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v4 = *v3;
  if ( *(_BYTE *)(*v3 + 16) || sub_15E4F60(*v3) )
    return 0x7FFFFFFF;
  v6 = (_QWORD *)(v2 + 56);
  if ( ((a2 >> 2) & 1) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v6, -1, 3) )
      goto LABEL_9;
    v8 = *(_QWORD *)(v2 - 24);
    if ( *(_BYTE *)(v8 + 16) )
      return 0x7FFFFFFF;
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v6, -1, 3) )
      goto LABEL_9;
    v8 = *(_QWORD *)(v2 - 72);
    if ( *(_BYTE *)(v8 + 16) )
      return 0x7FFFFFFF;
  }
  v9[0] = *(_QWORD *)(v8 + 112);
  if ( !(unsigned __int8)sub_1560260(v9, -1, 3) )
    return 0x7FFFFFFF;
LABEL_9:
  v7 = sub_3850BA0(v4);
  result = 0x80000000LL;
  if ( !v7 )
    return 0x7FFFFFFF;
  return result;
}
