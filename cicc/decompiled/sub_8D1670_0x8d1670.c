// Function: sub_8D1670
// Address: 0x8d1670
//
_BOOL8 __fastcall sub_8D1670(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // rdi
  _UNKNOWN *__ptr32 *v4; // r8
  _BOOL8 result; // rax
  _UNKNOWN *__ptr32 *v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx

  v3 = sub_730E00(a1);
  result = 0;
  if ( *(_BYTE *)(v3 + 173) == 12 )
  {
    v6 = (_UNKNOWN *__ptr32 *)qword_4F60578;
    result = 1;
    if ( qword_4F60578 )
    {
      if ( *(_BYTE *)(v3 + 176) == 1 )
      {
        result = 0;
        if ( !dword_4F60570 )
        {
          v7 = sub_72E9A0(v3);
          return (unsigned int)sub_73A2D0((__int64)v7, v6, v8, v9) != 0;
        }
      }
      else
      {
        return (unsigned int)sub_73A2C0(v3, qword_4F60578, v1, v2, v4) != 0;
      }
    }
  }
  return result;
}
