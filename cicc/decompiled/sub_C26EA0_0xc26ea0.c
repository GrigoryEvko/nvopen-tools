// Function: sub_C26EA0
// Address: 0xc26ea0
//
__int64 __fastcall sub_C26EA0(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  unsigned __int64 v3; // [rsp+0h] [rbp-30h] BYREF
  char v4; // [rsp+10h] [rbp-20h]

  sub_C22550((__int64)&v3, a1);
  if ( (v4 & 1) == 0 || (result = (unsigned int)v3, !(_DWORD)v3) )
  {
    if ( v3 )
    {
      v2 = 0;
      while ( 1 )
      {
        result = sub_C26D10(a1, v2);
        if ( (_DWORD)result )
          break;
        if ( v3 <= ++v2 )
          goto LABEL_8;
      }
    }
    else
    {
LABEL_8:
      sub_C1AFD0();
      return 0;
    }
  }
  return result;
}
