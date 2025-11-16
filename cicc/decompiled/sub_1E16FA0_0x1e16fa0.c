// Function: sub_1E16FA0
// Address: 0x1e16fa0
//
__int64 __fastcall sub_1E16FA0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r8

  result = a4;
  v8 = *(_QWORD *)(a1 + 32) + 40LL * (unsigned int)a2;
  if ( !*(_BYTE *)v8 && a3 == *(_DWORD *)(v8 + 8) )
    return sub_1E16EE0(a1, a2, result, a5, a6);
  return result;
}
