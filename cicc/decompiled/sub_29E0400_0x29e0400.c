// Function: sub_29E0400
// Address: 0x29e0400
//
__int64 __fastcall sub_29E0400(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 8);
  v2 = v1;
  if ( (unsigned int)*(unsigned __int8 *)(v1 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v1 + 16);
  v3 = *(_DWORD *)(v2 + 8) >> 8;
  v4 = sub_BCE3C0(*(__int64 **)v1, v3);
  if ( v4 == v1 )
    return sub_29E03A0(*(_QWORD *)(a1 + 16));
  v5 = *(_QWORD *)(a1 + 16);
  if ( !v5 )
    return 0;
  while ( 1 )
  {
    v6 = *(_QWORD *)(v5 + 24);
    if ( v4 == *(_QWORD *)(v6 + 8) && (unsigned __int8 *)a1 == sub_BD3990(*(unsigned __int8 **)(v5 + 24), v3) )
    {
      result = sub_29E03A0(*(_QWORD *)(v6 + 16));
      if ( (_BYTE)result )
        break;
    }
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      return 0;
  }
  return result;
}
