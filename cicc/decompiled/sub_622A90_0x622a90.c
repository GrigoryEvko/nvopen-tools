// Function: sub_622A90
// Address: 0x622a90
//
__int64 __fastcall sub_622A90(unsigned int a1, int a2)
{
  unsigned int v2; // r12d
  char v4; // bl
  __int64 i; // r15
  __int64 v6; // [rsp+8h] [rbp-48h]
  int v7; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 / dword_4F06BA0;
  if ( a1 == v6 * dword_4F06BA0 )
  {
    if ( !HIDWORD(qword_4F077B4)
      || (v2 = (unsigned __int8)((a2 == 0) + 5), sub_622920((unsigned int)(a2 == 0) + 5, v8, &v7), v8[0] != v6) )
    {
      v4 = 1;
      for ( i = 0; ; ++i )
      {
        v2 = i;
        sub_622920((unsigned int)i, v8, &v7);
        if ( v8[0] == v6 && byte_4B6DF90[i] == a2 )
        {
          if ( !HIDWORD(qword_4F077B4) || (_BYTE)i )
            return v2;
        }
        else if ( v4 == 13 )
        {
          return 13;
        }
        ++v4;
      }
    }
  }
  else
  {
    return 13;
  }
  return v2;
}
