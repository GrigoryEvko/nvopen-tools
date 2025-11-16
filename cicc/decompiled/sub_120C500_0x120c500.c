// Function: sub_120C500
// Address: 0x120c500
//
__int64 __fastcall sub_120C500(__int64 a1, __int64 a2, _BYTE *a3, _DWORD *a4, _DWORD *a5, unsigned __int8 *a6)
{
  int v10; // eax
  _DWORD *v11; // r10
  __int64 result; // rax
  unsigned __int64 v13; // rsi
  unsigned __int8 v14; // [rsp+Fh] [rbp-51h]
  const char *v15; // [rsp+10h] [rbp-50h] BYREF
  char v16; // [rsp+30h] [rbp-30h]
  char v17; // [rsp+31h] [rbp-2Fh]

  v10 = sub_1205440(*(_DWORD *)(a1 + 240), a3);
  *v11 = v10;
  if ( *a3 )
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  sub_120C370(a1, a6);
  sub_120C3D0(a1, a4);
  sub_120C4A0(a1, a5);
  result = *a6;
  if ( (_BYTE)result )
  {
    if ( *a5 == 1 )
    {
      v13 = *(_QWORD *)(a1 + 232);
      v14 = *a6;
      v15 = "dso_location and DLL-StorageClass mismatch";
      v17 = 1;
      v16 = 3;
      sub_11FD800(a1 + 176, v13, (__int64)&v15, 1);
      return v14;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
