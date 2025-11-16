// Function: sub_20C30B0
// Address: 0x20c30b0
//
__int64 __fastcall sub_20C30B0(unsigned int *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int v7; // r13d
  unsigned int v8; // eax
  unsigned int v9; // edx
  _BYTE *v10; // rsi
  unsigned int v12[13]; // [rsp+1Ch] [rbp-34h] BYREF

  result = *a1;
  v12[0] = 0;
  if ( (_DWORD)result )
  {
    v7 = 0;
    do
    {
      v8 = *(_DWORD *)(*((_QWORD *)a1 + 4) + 4LL * v7);
      do
      {
        v9 = v8;
        v8 = *(_DWORD *)(*((_QWORD *)a1 + 1) + 4LL * v8);
      }
      while ( v9 != v8 );
      if ( a2 == v8 && sub_20C2FE0(a4, v12) )
      {
        v10 = *(_BYTE **)(a3 + 8);
        if ( v10 == *(_BYTE **)(a3 + 16) )
        {
          sub_B8BBF0(a3, v10, v12);
        }
        else
        {
          if ( v10 )
          {
            *(_DWORD *)v10 = v7;
            v10 = *(_BYTE **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v10 + 4;
        }
      }
      result = v12[0];
      v7 = v12[0] + 1;
      v12[0] = v7;
    }
    while ( *a1 != v7 );
  }
  return result;
}
