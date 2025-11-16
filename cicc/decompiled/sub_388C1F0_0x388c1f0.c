// Function: sub_388C1F0
// Address: 0x388c1f0
//
__int64 __fastcall sub_388C1F0(__int64 a1, __int64 a2, _BYTE *a3, _DWORD *a4, _DWORD *a5, _BYTE *a6)
{
  int v10; // eax
  _DWORD *v11; // r10
  __int64 result; // rax
  unsigned __int64 v13; // rsi
  const char *v14; // [rsp+0h] [rbp-40h] BYREF
  char v15; // [rsp+10h] [rbp-30h]
  char v16; // [rsp+11h] [rbp-2Fh]

  v10 = sub_3887320(*(_DWORD *)(a1 + 64), a3);
  *v11 = v10;
  if ( *a3 )
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  sub_388C0E0(a1, a6);
  sub_388C130(a1, a4);
  sub_388C1A0(a1, a5);
  result = (unsigned __int8)*a6;
  if ( (_BYTE)result )
  {
    result = 0;
    if ( *a5 == 1 )
    {
      v13 = *(_QWORD *)(a1 + 56);
      v16 = 1;
      v15 = 3;
      v14 = "dso_location and DLL-StorageClass mismatch";
      return sub_38814C0(a1 + 8, v13, (__int64)&v14);
    }
  }
  return result;
}
