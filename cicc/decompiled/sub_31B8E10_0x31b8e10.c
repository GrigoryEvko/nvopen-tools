// Function: sub_31B8E10
// Address: 0x31b8e10
//
void __fastcall sub_31B8E10(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  _QWORD *v5; // r8
  __int64 v6; // rax
  _QWORD *v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // r8
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 i; // rcx
  size_t v13; // rdx
  _QWORD *v14; // r14
  __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 32);
  if ( v4 )
  {
    v5 = *(_QWORD **)v4;
    v6 = *(unsigned int *)(v4 + 8);
    v15[0] = a1;
    v7 = &v5[v6];
    v8 = sub_31B7EA0(v5, (__int64)v7, v15);
    v10 = v8;
    if ( v7 == v8 )
    {
      v14 = v7;
    }
    else
    {
      v11 = v8 + 1;
      if ( v7 == v11 )
      {
        v14 = v10;
      }
      else
      {
        for ( i = a1; ; i = v15[0] )
        {
          if ( *v11 != i )
            *v10++ = *v11;
          if ( v7 == ++v11 )
            break;
        }
        v9 = *(_QWORD **)v4;
        v13 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8) - (_QWORD)v7;
        v14 = (_QWORD *)((char *)v10 + v13);
        if ( v7 != (_QWORD *)(*(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8)) )
        {
          memmove(v10, v7, v13);
          v9 = *(_QWORD **)v4;
        }
      }
    }
    *(_DWORD *)(v4 + 8) = v14 - v9;
  }
  *(_QWORD *)(a1 + 32) = a2;
}
