// Function: sub_729AB0
// Address: 0x729ab0
//
__int64 __fastcall sub_729AB0(int a1)
{
  __int64 result; // rax
  unsigned int v2; // edi

  result = 0;
  if ( a1 )
  {
    result = unk_4F07280;
    if ( unk_4F07280 )
    {
      v2 = a1 - 1;
      if ( v2 > *(_DWORD *)(unk_4F07280 + 28LL) )
      {
        do
          result = *(_QWORD *)(result + 56);
        while ( *(_DWORD *)(result + 28) < v2 );
      }
    }
  }
  return result;
}
