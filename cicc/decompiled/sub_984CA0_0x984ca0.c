// Function: sub_984CA0
// Address: 0x984ca0
//
__int64 __fastcall sub_984CA0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx

  if ( !a1 )
    return 0;
  if ( !*(_QWORD *)(a1 + 40) )
    return 0;
  v1 = sub_AA54C0(*(_QWORD *)(a1 + 40));
  if ( !v1 )
    return 0;
  v2 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == v1 + 48 )
    goto LABEL_17;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_17:
    BUG();
  if ( *(_BYTE *)(v2 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  result = *(_QWORD *)(v2 - 120);
  if ( !result )
    return 0;
  v4 = *(_QWORD *)(v2 - 56);
  if ( !v4 )
    return 0;
  v5 = *(_QWORD *)(v2 - 88);
  if ( !v5 || v4 == v5 )
    return 0;
  return result;
}
