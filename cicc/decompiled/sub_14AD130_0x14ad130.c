// Function: sub_14AD130
// Address: 0x14ad130
//
__int64 __fastcall sub_14AD130(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  int v5; // eax
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // rax
  int v9; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = (a1 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  if ( (a1 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560490(v3, 38, &v9) && (v5 = v9) != 0
      || (v4 = *(_QWORD *)(v2 - 24), !*(_BYTE *)(v4 + 16))
      && (v10[0] = *(_QWORD *)(v4 + 112), (unsigned __int8)sub_1560490(v10, 38, &v9))
      && (v5 = v9) != 0 )
    {
      result = *(_QWORD *)(v2 + 24 * ((unsigned int)(v5 - 1) - (unsigned __int64)(*(_DWORD *)(v2 + 20) & 0xFFFFFFF)));
      if ( result )
        return result;
    }
    if ( sub_14AD0D0(a1) )
      return *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    return 0;
  }
  if ( (!(unsigned __int8)sub_1560490(v3, 38, &v9) || (v7 = v9) == 0)
    && ((v8 = *(_QWORD *)(v2 - 72), *(_BYTE *)(v8 + 16))
     || (v10[0] = *(_QWORD *)(v8 + 112), !(unsigned __int8)sub_1560490(v10, 38, &v9))
     || (v7 = v9) == 0)
    || (result = *(_QWORD *)(v2 + 24 * ((unsigned int)(v7 - 1) - (unsigned __int64)(*(_DWORD *)(v2 + 20) & 0xFFFFFFF)))) == 0 )
  {
    if ( sub_14AD0D0(a1) )
      return *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    return 0;
  }
  return result;
}
