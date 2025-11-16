// Function: sub_CFC2E0
// Address: 0xcfc2e0
//
__int64 __fastcall sub_CFC2E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rsi
  __int64 v8; // r14
  __int64 result; // rax
  _QWORD *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  int v14; // ebx
  __int64 v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v16, a6);
  result = *(_QWORD *)a1;
  v10 = (_QWORD *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v10 )
  {
    v11 = v8;
    do
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = 4;
        *(_QWORD *)(v11 + 8) = 0;
        v12 = *(_QWORD *)(result + 16);
        *(_QWORD *)(v11 + 16) = v12;
        LOBYTE(v7) = v12 != -4096;
        if ( ((v12 != 0) & (unsigned __int8)v7) != 0 && v12 != -8192 )
        {
          v15 = result;
          v7 = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050((unsigned __int64 *)v11, v7);
          result = v15;
        }
        *(_DWORD *)(v11 + 24) = *(_DWORD *)(result + 24);
      }
      result += 32;
      v11 += 32;
    }
    while ( v10 != (_QWORD *)result );
    v13 = *(_QWORD **)a1;
    v10 = (_QWORD *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v10 )
    {
      do
      {
        result = *(v10 - 2);
        v10 -= 4;
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
