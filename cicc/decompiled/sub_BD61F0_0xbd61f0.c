// Function: sub_BD61F0
// Address: 0xbd61f0
//
__int64 __fastcall sub_BD61F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // r14
  __int64 result; // rax
  _QWORD *v6; // r12
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // rdx
  _QWORD *v9; // rbx
  int v10; // ebx
  __int64 v11; // [rsp+8h] [rbp-48h]
  _QWORD v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = sub_C8D7D0(a1, a1 + 16, a2, 24, v12);
  result = *(_QWORD *)a1;
  v6 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v6 )
  {
    v7 = (unsigned __int64 *)v4;
    do
    {
      if ( v7 )
      {
        *v7 = 6;
        v7[1] = 0;
        v8 = *(_QWORD *)(result + 16);
        v7[2] = v8;
        LOBYTE(v3) = v8 != -4096;
        if ( ((v8 != 0) & (unsigned __int8)v3) != 0 && v8 != -8192 )
        {
          v11 = result;
          v3 = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(v7, v3);
          result = v11;
        }
      }
      result += 24;
      v7 += 3;
    }
    while ( v6 != (_QWORD *)result );
    v9 = *(_QWORD **)a1;
    result = 3LL * *(unsigned int *)(a1 + 8);
    v6 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v6 )
    {
      do
      {
        result = *(v6 - 1);
        v6 -= 3;
        if ( result != -4096 && result != 0 && result != -8192 )
          result = sub_BD60C0(v6);
      }
      while ( v6 != v9 );
      v6 = *(_QWORD **)a1;
    }
  }
  v10 = v12[0];
  if ( (_QWORD *)(a1 + 16) != v6 )
    result = _libc_free(v6, v3);
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 12) = v10;
  return result;
}
