// Function: sub_D23EA0
// Address: 0xd23ea0
//
__int64 __fastcall sub_D23EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // r12d
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx

  v7 = sub_B19060(a3 + 48, (__int64)&unk_4F86C48, a3, a4);
  if ( !(_BYTE)v7
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v5, v6)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F86C48, v9, v10)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v11, v12) )
  {
    return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82428, v13, v14) ^ 1;
  }
  return v7;
}
