// Function: sub_D84560
// Address: 0xd84560
//
char __fastcall sub_D84560(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // dl
  char result; // al
  _DWORD *v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // [rsp+0h] [rbp-40h]
  _BYTE v8[48]; // [rsp+10h] [rbp-30h] BYREF

  v7 = sub_D84370(a1, a2, a3, 0);
  result = v3;
  if ( v3 )
    return sub_D84450(a1, v7);
  v5 = *(_DWORD **)(a1 + 8);
  if ( v5 )
  {
    if ( *v5 == 2 )
    {
      v6 = sub_B491C0(a2);
      sub_B2EE70((__int64)v8, v6, 0);
      return v8[16];
    }
  }
  return result;
}
