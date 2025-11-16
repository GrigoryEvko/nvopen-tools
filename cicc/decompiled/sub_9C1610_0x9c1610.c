// Function: sub_9C1610
// Address: 0x9c1610
//
__int64 __fastcall sub_9C1610(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r14
  __int64 v4; // rsi
  __int64 v6; // r13
  __int64 result; // rax
  _QWORD *v8; // r12
  __int64 v9; // r15
  __int64 v10; // rdi
  _QWORD *v11; // r15
  int v12; // r15d
  __int64 v13; // [rsp+8h] [rbp-48h]
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (_QWORD *)(a1 + 16);
  v4 = a1 + 16;
  v6 = sub_C8D7D0(a1, a1 + 16, a2, 64, v14);
  result = *(_QWORD *)a1;
  v8 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
  if ( *(_QWORD **)a1 != v8 )
  {
    v9 = v6;
    do
    {
      while ( 1 )
      {
        if ( v9 )
        {
          *(_DWORD *)(v9 + 8) = 0;
          *(_QWORD *)v9 = v9 + 16;
          *(_DWORD *)(v9 + 12) = 12;
          if ( *(_DWORD *)(result + 8) )
            break;
        }
        result += 64;
        v9 += 64;
        if ( v8 == (_QWORD *)result )
          goto LABEL_7;
      }
      v4 = result;
      v10 = v9;
      v13 = result;
      v9 += 64;
      sub_9B68C0(v10, (char **)result);
      result = v13 + 64;
    }
    while ( v8 != (_QWORD *)(v13 + 64) );
LABEL_7:
    v11 = *(_QWORD **)a1;
    v8 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
    if ( *(_QWORD **)a1 != v8 )
    {
      do
      {
        v8 -= 8;
        result = (__int64)(v8 + 2);
        if ( (_QWORD *)*v8 != v8 + 2 )
          result = _libc_free(*v8, v4);
      }
      while ( v8 != v11 );
      v8 = *(_QWORD **)a1;
    }
  }
  v12 = v14[0];
  if ( v3 != v8 )
    result = _libc_free(v8, v4);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v12;
  return result;
}
