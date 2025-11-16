// Function: sub_2F60CF0
// Address: 0x2f60cf0
//
__int64 __fastcall sub_2F60CF0(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r13
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rsi

  result = *a1;
  v2 = *(unsigned int *)(*a1 + 72);
  if ( (_DWORD)v2 )
  {
    v3 = 8 * v2;
    v4 = 0;
    do
    {
      result = a1[16] + 8 * v4;
      if ( !*(_DWORD *)result && *(_BYTE *)(result + 56) )
      {
        if ( *(_BYTE *)(result + 57) )
        {
          v5 = *(_QWORD *)(*(_QWORD *)(*a1 + 64) + v4);
          *(_QWORD *)(v5 + 8) = 0;
          result = sub_2E0A600(*a1, v5);
        }
      }
      v4 += 8;
    }
    while ( v3 != v4 );
  }
  return result;
}
