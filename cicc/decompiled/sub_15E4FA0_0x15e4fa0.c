// Function: sub_15E4FA0
// Address: 0x15e4fa0
//
__int64 __fastcall sub_15E4FA0(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax
  char v3; // dl

  v1 = *(unsigned __int8 *)(a1 + 16);
  result = a1;
  if ( *(_BYTE *)(a1 + 16) && v1 != 3 )
  {
    if ( (unsigned int)(v1 - 1) > 1 )
    {
      return 0;
    }
    else
    {
      result = sub_164A820(*(_QWORD *)(a1 - 24));
      v3 = *(_BYTE *)(result + 16);
      if ( v3 )
      {
        if ( v3 != 3 )
          return 0;
      }
    }
  }
  return result;
}
