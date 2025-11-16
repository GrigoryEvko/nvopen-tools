// Function: sub_8CA0A0
// Address: 0x8ca0a0
//
__int64 __fastcall sub_8CA0A0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  _QWORD *v3; // r12
  __int64 v4; // r14
  __int64 v5; // r14

  if ( a2 )
  {
    sub_8C7090(6, a1);
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 32);
    if ( v5 )
    {
      sub_8C6400(6, a1);
      sub_8D0810(v5);
      *(_QWORD *)(a1 + 32) = 0;
    }
  }
  result = *(unsigned __int8 *)(a1 + 140);
  if ( (unsigned __int8)(result - 9) <= 2u )
  {
    result = sub_8D2490(a1);
    if ( (_DWORD)result )
      return (__int64)sub_8C9FB0(a1, a2);
  }
  else if ( (_BYTE)result == 2 )
  {
    result = *(unsigned __int8 *)(a1 + 161);
    if ( (result & 8) != 0 && (**(_BYTE **)(a1 + 176) & 1) != 0 )
    {
      v3 = *(_QWORD **)(a1 + 168);
      if ( (result & 0x10) != 0 )
        v3 = (_QWORD *)v3[12];
      while ( v3 )
      {
        if ( a2 )
        {
          result = (__int64)sub_8C7090(2, (__int64)v3);
        }
        else
        {
          v4 = v3[4];
          if ( v4 )
          {
            sub_8C6400(2, (__int64)v3);
            result = sub_8D0810(v4);
            v3[4] = 0;
          }
        }
        v3 = (_QWORD *)v3[15];
      }
    }
  }
  return result;
}
