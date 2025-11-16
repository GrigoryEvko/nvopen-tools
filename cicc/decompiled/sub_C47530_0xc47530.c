// Function: sub_C47530
// Address: 0xc47530
//
__int64 __fastcall sub_C47530(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  unsigned int v8; // eax
  __int64 result; // rax
  unsigned int v10; // r9d
  __int64 v12; // rbx
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  char v15; // [rsp-10h] [rbp-50h]
  __int64 v16; // [rsp-10h] [rbp-50h]
  __int64 v17; // [rsp+0h] [rbp-40h]
  unsigned int v18; // [rsp+Ch] [rbp-34h]

  while ( a4 > a5 )
  {
    v8 = a4;
    a4 = a5;
    a5 = v8;
    result = a2;
    a2 = a3;
    a3 = result;
  }
  if ( a4 )
  {
    v10 = a5 + 1;
    v12 = 0;
    v17 = a4;
    do
    {
      v13 = *(_QWORD *)(a2 + 8 * v12);
      v14 = a1;
      v15 = (_DWORD)v12++ != 0;
      a1 += 8;
      v18 = v10;
      sub_C46FF0(v14, a3, v13, 0, a5, v10, v15);
      result = v16;
      v10 = v18;
    }
    while ( v17 != v12 );
  }
  return result;
}
