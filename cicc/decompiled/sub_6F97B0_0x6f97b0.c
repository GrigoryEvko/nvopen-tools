// Function: sub_6F97B0
// Address: 0x6f97b0
//
__int64 __fastcall sub_6F97B0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx

  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 && *(_BYTE *)(a1 + 17) == 2 )
  {
    result = sub_8D3A70(*(_QWORD *)a1);
    if ( (_DWORD)result )
    {
      if ( *(_BYTE *)(a1 + 16) == 1 )
      {
        v6 = *(_QWORD *)(a1 + 144);
        result = *(unsigned __int8 *)(v6 + 24);
        if ( (_BYTE)result == 5 )
        {
          result = *(unsigned __int8 *)(*(_QWORD *)(v6 + 56) + 48LL);
          v6 = (unsigned int)(result - 3);
          if ( (unsigned __int8)(result - 3) <= 2u || (_BYTE)result == 1 )
            return sub_6F9770(a1, a2, v6, v3, v4, v5);
        }
        else if ( (_BYTE)result == 1 )
        {
          result = (unsigned int)*(unsigned __int8 *)(v6 + 56) - 105;
          if ( (unsigned __int8)(*(_BYTE *)(v6 + 56) - 105) <= 4u )
            return sub_6F9770(a1, a2, v6, v3, v4, v5);
        }
      }
    }
  }
  return result;
}
