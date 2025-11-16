// Function: sub_38DDB90
// Address: 0x38ddb90
//
__int64 __fastcall sub_38DDB90(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx

  result = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)result )
  {
    v3 = 16LL * (unsigned int)(result - 1);
    do
    {
      while ( 1 )
      {
        result = v3 + *(_QWORD *)(a2 + 16);
        if ( *(_BYTE *)result == 4 )
          break;
        v3 -= 16;
        if ( v3 == -16 )
          return result;
      }
      v3 -= 16;
      result = sub_38DDAF0(a1, *(unsigned int **)(result + 8));
    }
    while ( v3 != -16 );
  }
  return result;
}
