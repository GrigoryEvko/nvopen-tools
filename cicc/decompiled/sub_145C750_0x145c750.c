// Function: sub_145C750
// Address: 0x145c750
//
__int64 __fastcall sub_145C750(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 result; // rax
  int v6; // eax
  int v7; // [rsp+4h] [rbp-1Ch] BYREF
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = *a1;
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = v2 + 56;
  if ( (v1 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560490(v3, 38, &v7) )
    {
      v6 = v7;
      if ( v7 )
        return *(_QWORD *)(v2 + 24 * ((unsigned int)(v6 - 1) - (unsigned __int64)(*(_DWORD *)(v2 + 20) & 0xFFFFFFF)));
    }
    v4 = *(_QWORD *)(v2 - 24);
    result = 0;
    if ( *(_BYTE *)(v4 + 16) )
      return result;
LABEL_4:
    v8[0] = *(_QWORD *)(v4 + 112);
    if ( !(unsigned __int8)sub_1560490(v8, 38, &v7) )
      return 0;
    v6 = v7;
    if ( !v7 )
      return 0;
    return *(_QWORD *)(v2 + 24 * ((unsigned int)(v6 - 1) - (unsigned __int64)(*(_DWORD *)(v2 + 20) & 0xFFFFFFF)));
  }
  if ( (unsigned __int8)sub_1560490(v3, 38, &v7) )
  {
    v6 = v7;
    if ( v7 )
      return *(_QWORD *)(v2 + 24 * ((unsigned int)(v6 - 1) - (unsigned __int64)(*(_DWORD *)(v2 + 20) & 0xFFFFFFF)));
  }
  v4 = *(_QWORD *)(v2 - 72);
  result = 0;
  if ( !*(_BYTE *)(v4 + 16) )
    goto LABEL_4;
  return result;
}
