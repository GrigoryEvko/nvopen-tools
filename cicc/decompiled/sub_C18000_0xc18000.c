// Function: sub_C18000
// Address: 0xc18000
//
__int64 __fastcall sub_C18000(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rsi
  __int64 v6; // r14
  __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rdi
  int v14; // ebx
  __int64 v15; // [rsp+8h] [rbp-48h]
  _QWORD v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = a1 + 16;
  v6 = sub_C8D7D0(a1, a1 + 16, a2, 72, v16);
  result = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      while ( 1 )
      {
        if ( v9 )
        {
          v10 = *(_QWORD *)result;
          *(_DWORD *)(v9 + 16) = 0;
          *(_DWORD *)(v9 + 20) = 12;
          *(_QWORD *)v9 = v10;
          *(_QWORD *)(v9 + 8) = v9 + 24;
          if ( *(_DWORD *)(result + 16) )
            break;
        }
        result += 72;
        v9 += 72;
        if ( v8 == result )
          goto LABEL_7;
      }
      v4 = result + 8;
      v11 = v9 + 8;
      v15 = result;
      v9 += 72;
      sub_C15E20(v11, (char **)(result + 8));
      result = v15 + 72;
    }
    while ( v8 != v15 + 72 );
LABEL_7:
    v12 = *(_QWORD *)a1;
    result = 9LL * *(unsigned int *)(a1 + 8);
    v8 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v8 -= 72;
        v13 = *(_QWORD *)(v8 + 8);
        result = v8 + 24;
        if ( v13 != v8 + 24 )
          result = _libc_free(v13, v4);
      }
      while ( v8 != v12 );
      v8 = *(_QWORD *)a1;
    }
  }
  v14 = v16[0];
  if ( v3 != v8 )
    result = _libc_free(v8, v4);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v14;
  return result;
}
