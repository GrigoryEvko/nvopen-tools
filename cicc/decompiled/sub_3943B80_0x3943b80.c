// Function: sub_3943B80
// Address: 0x3943b80
//
__int64 __fastcall sub_3943B80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9

  result = sub_3940E90((__int64)a1, a2, a3, a4, a5, a6);
  if ( !(_DWORD)result )
  {
    result = sub_39439C0(a1, a2, v7, v8, v9, v10);
    if ( !(_DWORD)result )
    {
      sub_393D180((__int64)a1, a2, v11, v12, v13, v14);
      return 0;
    }
  }
  return result;
}
