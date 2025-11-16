// Function: sub_B1CA60
// Address: 0xb1ca60
//
__int64 __fastcall sub_B1CA60(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r8
  __int64 v4; // rax
  _QWORD *v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // r8
  _QWORD *v8; // rdi
  _QWORD *v9; // rax
  size_t v10; // rdx
  _QWORD *v11; // r12
  __int64 result; // rax
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD **)a1;
  v4 = *(unsigned int *)(a1 + 8);
  v13[0] = a2;
  v5 = &v3[v4];
  v6 = sub_B18540(v3, (__int64)v5, v13);
  if ( v5 == v6 )
  {
    v11 = v5;
  }
  else
  {
    v8 = v6;
    v9 = v6 + 1;
    if ( v5 == v9 )
    {
      v11 = v8;
    }
    else
    {
      do
      {
        if ( *v9 != v13[0] )
          *v8++ = *v9;
        ++v9;
      }
      while ( v5 != v9 );
      v7 = *(_QWORD **)a1;
      v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v5;
      v11 = (_QWORD *)((char *)v8 + v10);
      if ( v5 != (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
      {
        memmove(v8, v5, v10);
        v7 = *(_QWORD **)a1;
      }
    }
  }
  result = v11 - v7;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
