// Function: sub_2553CD0
// Address: 0x2553cd0
//
__int64 __fastcall sub_2553CD0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  char v4; // bl
  __int64 result; // rax
  int v6; // r14d
  int v7; // ebx
  int v8; // r15d
  int v9; // edx
  unsigned int v10; // edx
  char v11; // [rsp+Fh] [rbp-41h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v11 = sub_B2DCE0((__int64)a2);
  v4 = sub_B2D610((__int64)a2, 41);
  result = *(unsigned __int8 *)(**(_QWORD **)(a2[3] + 16LL) + 8LL);
  if ( v11 )
  {
    if ( v4 && (_BYTE)result == 7 )
    {
      *(_DWORD *)(a3 + 8) |= 0x70007u;
      return result;
    }
    *(_DWORD *)(a3 + 8) |= 0x10001u;
  }
  if ( (_BYTE)result == 7 && v4 )
    *(_DWORD *)(a3 + 8) |= 0x40004u;
  result = sub_250CB50(a1, 1);
  v6 = result;
  if ( v4 == 1 && (int)result >= 0 )
  {
    v12[0] = a2[15];
    result = sub_A74390(v12, 52, 0);
    if ( (_BYTE)result )
    {
      result = a2[13];
      v7 = 0;
      v8 = result;
      if ( (_DWORD)result )
      {
        while ( 1 )
        {
          result = sub_B2D640((__int64)a2, v7, 52);
          if ( (_BYTE)result )
            break;
          if ( v8 == ++v7 )
            return result;
        }
        v9 = *(_DWORD *)(a3 + 8);
        result = v9 | 0x70007u;
        v10 = v9 | 0x40004;
        if ( v6 == v7 )
        {
          *(_WORD *)(a3 + 10) = *(_WORD *)(a3 + 8) | *(_WORD *)(a3 + 10) & 0xFFFB;
        }
        else
        {
          if ( !v11 )
            result = v10;
          *(_DWORD *)(a3 + 8) = result;
        }
      }
    }
  }
  return result;
}
