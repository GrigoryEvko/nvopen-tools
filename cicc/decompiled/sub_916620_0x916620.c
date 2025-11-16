// Function: sub_916620
// Address: 0x916620
//
__int64 __fastcall sub_916620(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  char v4; // dl
  int v5; // edx

  result = sub_916430(a1, a2, a3);
  v4 = *(_BYTE *)result;
  if ( *(_BYTE *)result == 5 )
  {
    v5 = *(unsigned __int16 *)(result + 2);
    if ( v5 != 34 && v5 != 49 )
      sub_91B8A0("codegen error while generating initialization");
    result = *(_QWORD *)(result - 32LL * (*(_DWORD *)(result + 4) & 0x7FFFFFF));
    v4 = *(_BYTE *)result;
  }
  if ( v4 != 3 )
    return 0;
  return result;
}
