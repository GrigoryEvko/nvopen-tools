// Function: sub_B97700
// Address: 0xb97700
//
__int64 __fastcall sub_B97700(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rsi
  __int64 v4; // rax
  unsigned __int8 **v5; // rbx
  _QWORD *v6; // r14
  __int64 v7; // r12
  unsigned __int8 **v8; // rbx
  int v9; // ebx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (unsigned __int8 *)(a1 + 16);
  v4 = sub_C8D7D0(a1, a1 + 16, a2, 8, v12);
  v5 = *(unsigned __int8 ***)a1;
  v11 = v4;
  v6 = (_QWORD *)v4;
  v7 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    do
    {
      if ( v6 )
      {
        v3 = *v5;
        *v6 = *v5;
        if ( v3 )
          sub_B976B0((__int64)v5, v3, (__int64)v6);
        *v5 = 0;
      }
      ++v5;
      ++v6;
    }
    while ( (unsigned __int8 **)v7 != v5 );
    v8 = *(unsigned __int8 ***)a1;
    v7 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v3 = *(unsigned __int8 **)(v7 - 8);
        v7 -= 8;
        if ( v3 )
          sub_B91220(v7, (__int64)v3);
      }
      while ( (unsigned __int8 **)v7 != v8 );
      v7 = *(_QWORD *)a1;
    }
  }
  v9 = v12[0];
  if ( a1 + 16 != v7 )
    _libc_free(v7, v3);
  *(_DWORD *)(a1 + 12) = v9;
  *(_QWORD *)a1 = v11;
  return v11;
}
