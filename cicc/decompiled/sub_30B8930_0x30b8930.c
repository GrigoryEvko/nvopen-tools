// Function: sub_30B8930
// Address: 0x30b8930
//
__int64 __fastcall sub_30B8930(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r14
  __int64 v10; // rsi
  unsigned __int8 *v11; // r14
  __int64 v12; // rax

  if ( *a2 <= 0x1Cu || (unsigned __int8)(*a2 - 61) > 1u )
    BUG();
  result = 0;
  v6 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v6 == 63 )
  {
    v10 = *((_QWORD *)a2 - 4);
    sub_30B8700(a1, v10, a4, a5);
    if ( *(_DWORD *)(a5 + 8)
      && *(_DWORD *)(a4 + 8) > 1u
      && (v11 = sub_BD3990(*(unsigned __int8 **)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)), v10),
          v12 = sub_D97190(a1, a3),
          *(_WORD *)(v12 + 24) == 15)
      && v11 == *(unsigned __int8 **)(v12 - 8) )
    {
      return 1;
    }
    else
    {
      *(_DWORD *)(a4 + 8) = 0;
      return 0;
    }
  }
  return result;
}
