// Function: sub_76D840
// Address: 0x76d840
//
__int64 __fastcall sub_76D840(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax

  if ( a1 )
  {
    v2 = a1;
    do
    {
      sub_76C8B0(v2, a2);
      result = *(unsigned int *)(a2 + 72);
      if ( (_DWORD)result )
        break;
      v2 = *(_QWORD *)(v2 + 16);
    }
    while ( v2 );
  }
  return result;
}
