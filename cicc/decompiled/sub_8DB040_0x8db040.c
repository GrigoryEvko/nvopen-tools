// Function: sub_8DB040
// Address: 0x8db040
//
__int64 __fastcall sub_8DB040(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  char v4; // dl
  __int64 v5; // rcx
  __int64 v6; // [rsp+0h] [rbp-10h] BYREF
  __int64 v7; // [rsp+8h] [rbp-8h] BYREF

  v7 = a1;
  v6 = a2;
  result = sub_8DAE50(a1, a2, &v7, &v6);
  if ( (_DWORD)result )
  {
    v4 = *(_BYTE *)(v7 + 140);
    if ( v4 )
    {
      v5 = *(unsigned __int8 *)(v6 + 140);
      if ( (_BYTE)v5 )
      {
        result = 0;
        if ( v4 == (_BYTE)v5 )
        {
          if ( v4 == 2 || v4 == 3 )
          {
            return *(_QWORD *)(v7 + 128) == *(_QWORD *)(v6 + 128);
          }
          else
          {
            result = 1;
            if ( v7 != v6 )
              return (unsigned int)sub_8D97D0(v7, v6, 0x2000u, v5, v3) != 0;
          }
        }
      }
    }
  }
  return result;
}
