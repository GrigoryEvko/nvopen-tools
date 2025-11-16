// Function: sub_2241C20
// Address: 0x2241c20
//
__int64 __fastcall sub_2241C20(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  unsigned int v5; // r8d
  __int64 v6; // rdx

  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL))(a1);
  v5 = 0;
  if ( *(_QWORD *)(a3 + 8) == v6 )
    LOBYTE(v5) = *(_DWORD *)a3 == v4;
  return v5;
}
