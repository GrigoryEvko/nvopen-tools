// Function: sub_2A6CD80
// Address: 0x2a6cd80
//
__int64 __fastcall sub_2A6CD80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  unsigned int v5; // r12d
  __int64 v6; // rbx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 i; // rbx
  unsigned __int8 *v11; // rsi
  int v12; // eax
  __int64 v14; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v14 = a2 + 72;
  if ( v4 != a2 + 72 )
  {
    v5 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = v4 - 24;
        if ( !v4 )
          v6 = 0;
        if ( *(_BYTE *)(a1 + 68) )
          break;
        if ( sub_C8CA60(a1 + 40, v6) )
          goto LABEL_10;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v14 == v4 )
          return v5;
      }
      v7 = *(_QWORD **)(a1 + 48);
      v8 = (__int64)&v7[*(unsigned int *)(a1 + 60)];
      if ( v7 != (_QWORD *)v8 )
      {
        while ( v6 != *v7 )
        {
          if ( (_QWORD *)v8 == ++v7 )
            goto LABEL_14;
        }
LABEL_10:
        v9 = *(_QWORD *)(v6 + 56);
        for ( i = v6 + 48; i != v9; v5 |= v12 )
        {
          v11 = (unsigned __int8 *)(v9 - 24);
          if ( !v9 )
            v11 = 0;
          v12 = sub_2A6CBA0(a1, v11, v8, a4);
          v9 = *(_QWORD *)(v9 + 8);
        }
      }
LABEL_14:
      v4 = *(_QWORD *)(v4 + 8);
      if ( v14 == v4 )
        return v5;
    }
  }
  return 0;
}
