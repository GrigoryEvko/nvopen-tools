// Function: sub_778FE0
// Address: 0x778fe0
//
__int64 __fastcall sub_778FE0(__int64 a1, int a2, unsigned __int64 a3, __int64 a4)
{
  unsigned int v4; // esi
  int v6; // ebx
  char v7; // al
  __int64 result; // rax
  unsigned int v9; // edx
  int v10; // ecx
  int v11; // esi
  int v12[5]; // [rsp+1Ch] [rbp-14h] BYREF

  v4 = a2 - a4;
  v6 = (v4 >> 3) + 10;
  v7 = *(_BYTE *)(a3 + 140);
  if ( (unsigned __int8)(v7 - 9) <= 2u || v7 == 8 )
  {
    v12[0] = 1;
    result = sub_7764B0(a1, a3, v12);
    v9 = result;
    if ( (_DWORD)result )
    {
      v10 = v4 & 7;
      do
      {
        result = a4 + -v6;
        if ( v10 || v9 <= 7 )
        {
          v11 = 1 << v10++;
          *(_BYTE *)result |= v11;
          if ( v10 == 8 )
          {
            ++v6;
            v10 = 0;
          }
          --v9;
        }
        else
        {
          *(_BYTE *)result = -1;
          ++v6;
          v9 -= 8;
        }
      }
      while ( v9 );
    }
  }
  else
  {
    result = (unsigned int)(1 << (v4 & 7));
    *(_BYTE *)(a4 + -v6) |= result;
  }
  return result;
}
