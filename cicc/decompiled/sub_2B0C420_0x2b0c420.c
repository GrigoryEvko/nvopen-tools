// Function: sub_2B0C420
// Address: 0x2b0c420
//
__int64 __fastcall sub_2B0C420(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r8
  __int64 v9; // rcx
  __int64 v10; // r11

  result = a2 - 1;
  v7 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v8 = (__int64 *)(a1 + 8 * v7);
      result = *v8;
      v9 = *(_QWORD *)(*v8 + 184);
      if ( v9 && (v10 = *(_QWORD *)(a4 + 184)) != 0 )
      {
        if ( *(_DWORD *)(v9 + 200) >= *(_DWORD *)(v10 + 200) )
          break;
      }
      else if ( *(_DWORD *)(result + 200) >= *(_DWORD *)(a4 + 200) )
      {
        break;
      }
      *(_QWORD *)(a1 + 8 * a2) = result;
      a2 = v7;
      result = (v7 - 1) / 2;
      if ( a3 >= v7 )
        goto LABEL_10;
      v7 = (v7 - 1) / 2;
    }
  }
  v8 = (__int64 *)(a1 + 8 * a2);
LABEL_10:
  *v8 = a4;
  return result;
}
