// Function: sub_264C6F0
// Address: 0x264c6f0
//
__int64 __fastcall sub_264C6F0(__int64 a1, __int64 a2)
{
  int *v2; // rbx
  int *v3; // r12
  __int64 result; // rax
  _BYTE v5[80]; // [rsp+0h] [rbp-50h] BYREF

  v2 = *(int **)(a2 + 8);
  v3 = &v2[*(unsigned int *)(a2 + 24)];
  result = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)result && v3 != v2 )
  {
    while ( (unsigned int)*v2 > 0xFFFFFFFD )
    {
      if ( v3 == ++v2 )
        return result;
    }
    if ( v3 != v2 )
    {
LABEL_8:
      result = sub_22B6470((__int64)v5, a1, v2);
      while ( v3 != ++v2 )
      {
        if ( (unsigned int)*v2 <= 0xFFFFFFFD )
        {
          if ( v3 != v2 )
            goto LABEL_8;
          return result;
        }
      }
    }
  }
  return result;
}
