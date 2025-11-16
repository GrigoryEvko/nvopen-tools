// Function: sub_3940DC0
// Address: 0x3940dc0
//
__int64 __fastcall sub_3940DC0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD v12[2]; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+10h] [rbp-40h]
  _QWORD v14[2]; // [rsp+20h] [rbp-30h] BYREF
  char v15; // [rsp+30h] [rbp-20h]

  v1 = a1[6];
  a1[9] = *(_QWORD *)(v1 + 8);
  a1[10] = *(_QWORD *)(v1 + 16);
  sub_393FF90((__int64)v12, a1);
  if ( (v13 & 1) == 0 || (result = LODWORD(v12[0]), v2 = v12[1], !LODWORD(v12[0])) )
  {
    result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 32LL))(a1, v12[0], v2);
    if ( !(_DWORD)result )
    {
      sub_393FF90((__int64)v14, a1);
      if ( (v15 & 1) == 0 || (result = LODWORD(v14[0]), v4 = v14[1], !LODWORD(v14[0])) )
      {
        if ( v14[0] == 103 )
        {
          result = sub_3940A30(a1);
          if ( !(_DWORD)result )
          {
            result = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 40LL))(a1);
            if ( !(_DWORD)result )
            {
              sub_393D180((__int64)a1, (__int64)a1, v8, v9, v10, v11);
              return 0;
            }
          }
        }
        else
        {
          sub_393D180((__int64)v14, (__int64)a1, v4, v5, v6, v7);
          return 2;
        }
      }
    }
  }
  return result;
}
