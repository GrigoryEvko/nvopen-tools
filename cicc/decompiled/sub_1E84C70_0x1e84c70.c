// Function: sub_1E84C70
// Address: 0x1e84c70
//
__int64 __fastcall sub_1E84C70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // r13

  v6 = 11LL * *(int *)(a2 + 48);
  v7 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(a2 + 48);
  if ( *(_DWORD *)(v7 + 24) != -1 && *(_DWORD *)(v7 + 28) != -1 )
  {
    if ( *(_BYTE *)(v7 + 32) )
      goto LABEL_4;
LABEL_7:
    sub_1E832A0(a1, a2, v6, a4, a5, a6);
    if ( *(_BYTE *)(v7 + 33) )
      return a1;
LABEL_8:
    sub_1E83950(a1, a2, v6, a4, a5, a6);
    return a1;
  }
  sub_1E81750(a1, a2);
  if ( !*(_BYTE *)(v7 + 32) )
    goto LABEL_7;
LABEL_4:
  if ( !*(_BYTE *)(v7 + 33) )
    goto LABEL_8;
  return a1;
}
