// Function: sub_D04070
// Address: 0xd04070
//
__int64 __fastcall sub_D04070(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 *v3; // r10
  __int64 v4; // r11
  __int64 v5; // r8
  bool v6; // r9
  __int64 *v7; // r10
  __int64 v8; // r11
  __int64 result; // rax
  unsigned int v10; // eax
  unsigned int v11; // eax

  if ( sub_D002E0(a2, 153) )
  {
    v11 = sub_D03F50(v4, v2, v3);
    return (((unsigned __int8)(v11 >> 6) | (unsigned __int8)((v11 >> 4) | v11 | (v11 >> 2))) & 2) != 0;
  }
  else
  {
    v6 = sub_D002E0(v2, 153);
    result = 3;
    if ( v6 )
    {
      v10 = sub_D03F50(v8, v5, v7);
      result = ((unsigned __int8)(v10 >> 6) | (unsigned __int8)((v10 >> 4) | v10 | (v10 >> 2))) & 2;
      if ( (_DWORD)result )
        return 2;
    }
  }
  return result;
}
