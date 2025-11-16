// Function: sub_15941C0
// Address: 0x15941c0
//
__int64 __fastcall sub_15941C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  int v7; // eax
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 *v10; // rdi
  __int64 v11; // rsi
  unsigned __int64 v12; // rcx
  __int64 v13; // rcx

  sub_1648CB0(a1, a2, a3);
  *(_DWORD *)(a1 + 20) = a5 & 0xFFFFFFF | *(_DWORD *)(a1 + 20) & 0xF0000000;
  v7 = a5;
  a5 *= 8;
  v8 = a5 >> 3;
  result = 24LL * (v7 & 0xFFFFFFF);
  v10 = (__int64 *)(a1 - result);
  if ( a5 > 0 )
  {
    do
    {
      result = *a4;
      if ( *v10 )
      {
        v11 = v10[1];
        v12 = v10[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v12 = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
      }
      *v10 = result;
      if ( result )
      {
        v13 = *(_QWORD *)(result + 8);
        v10[1] = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = (unsigned __int64)(v10 + 1) | *(_QWORD *)(v13 + 16) & 3LL;
        v10[2] = (result + 8) | v10[2] & 3;
        *(_QWORD *)(result + 8) = v10;
      }
      ++a4;
      v10 += 3;
      --v8;
    }
    while ( v8 );
  }
  return result;
}
