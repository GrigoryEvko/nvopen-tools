// Function: sub_76D3C0
// Address: 0x76d3c0
//
__int64 __fastcall sub_76D3C0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // rbx
  __int64 result; // rax

  if ( a1 )
  {
    v5 = a1;
    do
    {
      sub_76CDC0(v5, a2, a3, a4, a5);
      result = *(unsigned int *)(a2 + 72);
      if ( (_DWORD)result )
        break;
      v5 = (_QWORD *)v5[2];
    }
    while ( v5 );
  }
  return result;
}
