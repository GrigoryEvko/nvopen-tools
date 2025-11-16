// Function: sub_B979A0
// Address: 0xb979a0
//
void __fastcall sub_B979A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdx
  int v5; // eax
  __int64 *v6; // rdi
  unsigned __int8 *v7; // rsi
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 *v10; // rdi
  unsigned __int8 **v11; // r13
  __int64 v12; // r12
  _QWORD *v13; // r15
  unsigned __int8 **v14; // r13
  int v15; // r13d
  _QWORD v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 56);
  v4 = *(unsigned int *)(v3 + 8);
  if ( *(_DWORD *)(v3 + 12) <= (unsigned int)v4 )
  {
    v7 = (unsigned __int8 *)(v3 + 16);
    v8 = sub_C8D7D0(*(_QWORD *)(a1 + 56), v3 + 16, 0, 8, v16);
    v9 = 8LL * *(unsigned int *)(v3 + 8);
    v10 = (__int64 *)(v9 + v8);
    if ( v9 + v8 )
    {
      *v10 = a2;
      if ( a2 )
      {
        v7 = (unsigned __int8 *)a2;
        sub_B96E90((__int64)v10, a2, 1);
      }
      v9 = 8LL * *(unsigned int *)(v3 + 8);
    }
    v11 = *(unsigned __int8 ***)v3;
    v12 = *(_QWORD *)v3 + v9;
    if ( *(_QWORD *)v3 != v12 )
    {
      v13 = (_QWORD *)v8;
      do
      {
        if ( v13 )
        {
          v7 = *v11;
          *v13 = *v11;
          if ( v7 )
          {
            sub_B976B0((__int64)v11, v7, (__int64)v13);
            *v11 = 0;
          }
        }
        ++v11;
        ++v13;
      }
      while ( (unsigned __int8 **)v12 != v11 );
      v14 = *(unsigned __int8 ***)v3;
      v12 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
      if ( *(_QWORD *)v3 != v12 )
      {
        do
        {
          v7 = *(unsigned __int8 **)(v12 - 8);
          v12 -= 8;
          if ( v7 )
            sub_B91220(v12, (__int64)v7);
        }
        while ( (unsigned __int8 **)v12 != v14 );
        v12 = *(_QWORD *)v3;
      }
    }
    v15 = v16[0];
    if ( v3 + 16 != v12 )
      _libc_free(v12, v7);
    ++*(_DWORD *)(v3 + 8);
    *(_QWORD *)v3 = v8;
    *(_DWORD *)(v3 + 12) = v15;
  }
  else
  {
    v5 = *(_DWORD *)(v3 + 8);
    v6 = (__int64 *)(*(_QWORD *)v3 + 8 * v4);
    if ( v6 )
    {
      *v6 = a2;
      if ( a2 )
        sub_B96E90((__int64)v6, a2, 1);
      v5 = *(_DWORD *)(v3 + 8);
    }
    *(_DWORD *)(v3 + 8) = v5 + 1;
  }
}
