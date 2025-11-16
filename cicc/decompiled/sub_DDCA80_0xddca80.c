// Function: sub_DDCA80
// Address: 0xddca80
//
__int64 __fastcall sub_DDCA80(__int64 *a1, __int64 a2, _BYTE *a3, _BYTE *a4, __int64 a5)
{
  __int64 result; // rax
  unsigned int v9; // eax
  char v10; // r8

  result = sub_DC3AF0((__int64)a1, a2, a3, a4);
  if ( !BYTE1(result) )
  {
    if ( (unsigned __int8)sub_DDC560(a1, *(_QWORD *)(a5 + 40), a2, (__int64)a3, (__int64)a4) )
    {
      return 257;
    }
    else
    {
      v9 = sub_B52870(a2);
      v10 = sub_DDC560(a1, *(_QWORD *)(a5 + 40), ((unsigned __int64)BYTE4(a2) << 32) | v9, (__int64)a3, (__int64)a4);
      result = 0;
      if ( v10 )
        return 256;
    }
  }
  return result;
}
