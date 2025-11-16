// Function: sub_390E130
// Address: 0x390e130
//
__int64 __fastcall sub_390E130(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // rdx

  result = sub_390E0F0(a2, a3);
  if ( *(_BYTE *)(a1 + 24) )
  {
    v4 = *(_QWORD *)(a2 + 256);
    if ( !*(_BYTE *)(a2 + 248) )
      v4 = *(unsigned int *)(v4 + 72);
    return v4 + result - 1;
  }
  return result;
}
