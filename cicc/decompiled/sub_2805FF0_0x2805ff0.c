// Function: sub_2805FF0
// Address: 0x2805ff0
//
unsigned __int64 __fastcall sub_2805FF0(__int64 **a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // r15
  int v5; // r13d
  unsigned int v6; // ebx
  unsigned int v7; // esi
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result != a2 + 48 )
  {
    if ( !result )
      BUG();
    v3 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v3);
      v5 = result;
      if ( (_DWORD)result )
      {
        v6 = 0;
        do
        {
          v7 = v6++;
          v8 = (__int64 *)sub_B46EC0(v3, v7);
          result = (unsigned __int64)sub_2805C60(*a1, a2, v8, v9, v10, v11);
        }
        while ( v5 != v6 );
      }
    }
  }
  return result;
}
