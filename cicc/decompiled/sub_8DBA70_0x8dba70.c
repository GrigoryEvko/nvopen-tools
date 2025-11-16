// Function: sub_8DBA70
// Address: 0x8dba70
//
__int64 __fastcall sub_8DBA70(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  unsigned int v3; // r8d
  __int64 v5; // rdi
  __int64 v6; // rsi

  v2 = *(__int64 **)a2;
  v3 = 0;
  if ( *(_QWORD *)a1
    && v2
    && (*(_BYTE *)(a1 + 89) & 4) != 0
    && (*(_BYTE *)(a2 + 89) & 4) != 0
    && **(_QWORD **)a1 == *v2
    && (v3 = 1, v5 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v6 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL), v5 != v6) )
  {
    return (unsigned int)sub_8D97D0(v5, v6, 0, *v2, 1) != 0;
  }
  else
  {
    return v3;
  }
}
