// Function: sub_1277310
// Address: 0x1277310
//
__int64 __fastcall sub_1277310(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  char v4; // dl
  int v5; // edx

  result = sub_1277140(a1, a2, a3);
  v4 = *(_BYTE *)(result + 16);
  if ( v4 == 5 )
  {
    v5 = *(unsigned __int16 *)(result + 18);
    if ( v5 != 32 && v5 != 47 )
      sub_127B550("codegen error while generating initialization");
    result = *(_QWORD *)(result - 24LL * (*(_DWORD *)(result + 20) & 0xFFFFFFF));
    v4 = *(_BYTE *)(result + 16);
  }
  if ( v4 != 3 )
    return 0;
  return result;
}
