// Function: sub_31B7F60
// Address: 0x31b7f60
//
__int64 __fastcall sub_31B7F60(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  _QWORD *v3; // r9
  __int64 v4; // rax
  _QWORD *v5; // rsi
  _QWORD *v6; // rax
  __int64 v7; // r8
  _QWORD *v8; // r9
  _QWORD *v9; // rdi
  _QWORD *v10; // rax
  size_t v11; // rdx
  _QWORD *v12; // r12
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = a1[4];
  result = (__int64)&unk_4A34A00;
  *a1 = &unk_4A34A00;
  if ( v1 )
  {
    v3 = *(_QWORD **)v1;
    v4 = *(unsigned int *)(v1 + 8);
    v13[0] = (__int64)a1;
    v5 = &v3[v4];
    v6 = sub_31B7EA0(v3, (__int64)v5, v13);
    v9 = v6;
    if ( v5 == v6 )
    {
      v12 = v5;
    }
    else
    {
      v10 = v6 + 1;
      if ( v5 == v10 )
      {
        v12 = v9;
      }
      else
      {
        while ( 1 )
        {
          if ( *v10 != v7 )
            *v9++ = *v10;
          if ( v5 == ++v10 )
            break;
          v7 = v13[0];
        }
        v8 = *(_QWORD **)v1;
        v11 = *(_QWORD *)v1 + 8LL * *(unsigned int *)(v1 + 8) - (_QWORD)v5;
        v12 = (_QWORD *)((char *)v9 + v11);
        if ( v5 != (_QWORD *)(*(_QWORD *)v1 + 8LL * *(unsigned int *)(v1 + 8)) )
        {
          memmove(v9, v5, v11);
          v8 = *(_QWORD **)v1;
        }
      }
    }
    result = v12 - v8;
    *(_DWORD *)(v1 + 8) = result;
  }
  return result;
}
