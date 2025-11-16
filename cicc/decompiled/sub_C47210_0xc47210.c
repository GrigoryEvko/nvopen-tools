// Function: sub_C47210
// Address: 0xc47210
//
__int64 __fastcall sub_C47210(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rbx
  unsigned int v8; // r13d
  unsigned __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // r9d
  char v13; // [rsp-10h] [rbp-50h]
  __int64 v14; // [rsp+0h] [rbp-40h]

  if ( a4 )
  {
    v14 = a4;
    v7 = 0;
    v8 = 0;
    do
    {
      v9 = *(_QWORD *)(a3 + 8 * v7);
      v10 = a1;
      v13 = (_DWORD)v7 != 0;
      v11 = a4 - v7++;
      a1 += 8;
      v8 |= sub_C46FF0(v10, a2, v9, 0, a4, v11, v13);
    }
    while ( v7 != v14 );
  }
  else
  {
    return 0;
  }
  return v8;
}
