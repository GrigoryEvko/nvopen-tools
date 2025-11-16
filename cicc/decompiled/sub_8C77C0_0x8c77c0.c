// Function: sub_8C77C0
// Address: 0x8c77c0
//
__int64 __fastcall sub_8C77C0(__int64 a1)
{
  _BOOL4 v1; // r13d
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  _UNKNOWN *__ptr32 *v12; // r8

  v1 = 1;
  v2 = *(__int64 **)(a1 + 32);
  if ( !v2 )
    return v1;
  v3 = *v2;
  v4 = a1;
  if ( a1 != *v2 || (v4 = v2[1]) != 0 && v3 != v4 )
  {
    v1 = sub_8C7610(v4);
    if ( !v1 )
      return v1;
    v7 = *(_QWORD *)(v4 + 128);
    v8 = *(_QWORD *)(v3 + 128);
    if ( v7 == v8 )
    {
      v9 = *(_QWORD *)(v3 + 128);
    }
    else
    {
      if ( !(unsigned int)sub_8D97D0(v7, v8, 0, v5, v6) )
      {
LABEL_9:
        v1 = 0;
        sub_8C6700((__int64 *)v4, (unsigned int *)(v3 + 64), 0x42Au, 0x425u);
        sub_8C7090(2, v4);
        return v1;
      }
      v8 = *(_QWORD *)(v3 + 128);
      v9 = *(_QWORD *)(v4 + 128);
    }
    if ( (unsigned int)sub_8DBAE0(v9, v8)
      && (unsigned int)sub_73A2C0(v4, v3, v10, v11, v12)
      && ((*(_BYTE *)(v3 + 88) ^ *(_BYTE *)(v4 + 88)) & 0x73) == 0 )
    {
      return v1;
    }
    goto LABEL_9;
  }
  return 1;
}
