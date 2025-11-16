// Function: sub_1F3D840
// Address: 0x1f3d840
//
__int64 __fastcall sub_1F3D840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char (__fastcall *v5)(__int64, __int64, __int64); // rax
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // rax

  v5 = *(char (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 736LL);
  if ( v5 != sub_1F3CC60 )
    return -(((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v5)(
               a1,
               a2,
               a3,
               a4,
               a5,
               0)
           ^ 1);
  v6 = *(_QWORD *)(a3 + 8);
  result = 0xFFFFFFFFLL;
  if ( (unsigned __int64)(v6 + 0xFFFF) <= 0x1FFFD && !*(_QWORD *)a3 )
  {
    v8 = *(_QWORD *)(a3 + 24);
    if ( v8 == 1 )
    {
      return (unsigned int)-(*(_BYTE *)(a3 + 16) & (v6 != 0));
    }
    else if ( v8 == 2 )
    {
      return (unsigned int)-(unsigned __int8)(*(_BYTE *)(a3 + 16) | (v6 != 0));
    }
    else
    {
      return (unsigned int)-(v8 != 0);
    }
  }
  return result;
}
