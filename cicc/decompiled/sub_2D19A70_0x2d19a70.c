// Function: sub_2D19A70
// Address: 0x2d19a70
//
unsigned __int64 __fastcall sub_2D19A70(__int64 a1, unsigned int a2, int a3, __int64 a4)
{
  unsigned __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD v9[6]; // [rsp+0h] [rbp-30h] BYREF

  result = a2;
  if ( !a2 )
    result = 1LL << sub_AE5260(a1, a4);
  if ( (unsigned int)result <= 0xF && (((_BYTE)result - 1) & 0x10) == 0 )
  {
    v7 = sub_9208B0(a1, a4);
    v9[1] = v8;
    v9[0] = (unsigned __int64)(v7 + 7) >> 3;
    result = a3 * (unsigned int)sub_CA1930(v9);
    if ( (unsigned int)result > 0xF )
    {
      return 16;
    }
    else if ( !(_DWORD)result || ((unsigned int)result & ((_DWORD)result - 1)) != 0 )
    {
      return ((unsigned int)(((result >> 1) | result) >> 2) | (unsigned int)(result >> 1) | (unsigned int)result) + 1;
    }
  }
  return result;
}
