// Function: sub_D6B130
// Address: 0xd6b130
//
__int64 __fastcall sub_D6B130(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rsi
  __int64 v8; // r14
  __int64 result; // rax
  _QWORD *v10; // r12
  unsigned __int64 *v11; // rbx
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rbx
  int v14; // ebx
  __int64 v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v16, a6);
  result = *(_QWORD *)a1;
  v10 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v10 )
  {
    v11 = (unsigned __int64 *)v8;
    do
    {
      if ( v11 )
      {
        *v11 = 4;
        v11[1] = 0;
        v12 = *(_QWORD *)(result + 16);
        v11[2] = v12;
        LOBYTE(v7) = v12 != -4096;
        if ( ((v12 != 0) & (unsigned __int8)v7) != 0 && v12 != -8192 )
        {
          v15 = result;
          v7 = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(v11, v7);
          result = v15;
        }
      }
      result += 24;
      v11 += 3;
    }
    while ( v10 != (_QWORD *)result );
    v13 = *(_QWORD **)a1;
    result = 3LL * *(unsigned int *)(a1 + 8);
    v10 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v10 )
    {
      do
      {
        result = *(v10 - 1);
        v10 -= 3;
        if ( result != -4096 && result != 0 && result != -8192 )
          result = sub_BD60C0(v10);
      }
      while ( v10 != v13 );
      v10 = *(_QWORD **)a1;
    }
  }
  v14 = v16[0];
  if ( (_QWORD *)(a1 + 16) != v10 )
    result = _libc_free(v10, v7);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v14;
  return result;
}
