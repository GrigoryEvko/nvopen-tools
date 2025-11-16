// Function: sub_E92290
// Address: 0xe92290
//
__int64 __fastcall sub_E92290(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // eax
  __int64 v5; // [rsp+8h] [rbp-18h]

  v2 = a2;
  v5 = sub_E92200(a1, a2, 1);
  if ( BYTE4(v5) )
  {
    v3 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 16LL))(a1, (unsigned int)v5, 0);
    if ( v3 != -1 )
      return v3;
  }
  return v2;
}
