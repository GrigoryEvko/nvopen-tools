// Function: sub_D49500
// Address: 0xd49500
//
__int64 __fastcall sub_D49500(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx

  v7 = sub_B19060(a3 + 48, (__int64)&unk_4F875F0, a3, a4);
  if ( !(_BYTE)v7
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v5, v6)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F875F0, v9, v10)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v11, v12)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82420, v13, v14)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v15, v16) )
  {
    return (unsigned int)sub_B19060(a3, (__int64)&unk_4F82408, v17, v18) ^ 1;
  }
  return v7;
}
