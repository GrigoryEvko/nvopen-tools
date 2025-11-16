// Function: sub_34C9B30
// Address: 0x34c9b30
//
_QWORD *__fastcall sub_34C9B30(_QWORD **a1, __int64 a2, __int64 a3)
{
  _QWORD **v3; // rcx
  __int64 v5; // rbx
  _QWORD *v6; // r15
  _QWORD *result; // rax
  _QWORD *v8; // r13
  __int64 *v9; // r14
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // [rsp+8h] [rbp-38h]

  v3 = a1;
  v5 = a2;
  v6 = *a1;
  if ( a2 != a3 )
  {
    while ( !(unsigned __int8)sub_34C9970(v5) )
    {
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)v5 & 4) != 0 )
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( a3 == v5 )
          break;
      }
      else
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
        v5 = *(_QWORD *)(v5 + 8);
        if ( a3 == v5 )
          break;
      }
    }
  }
  result = v3[1];
  v8 = (_QWORD *)result[1];
  if ( a3 != v5 )
  {
    v9 = (__int64 *)(*result + 40LL);
    do
    {
      v12 = (__int64)sub_2E7B2C0(v6, v5);
      sub_2E31040(v9, v12);
      v10 = *(_QWORD *)v12;
      v11 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v12 + 8) = v8;
      *(_QWORD *)v12 = v11 | v10 & 7;
      *(_QWORD *)(v11 + 8) = v12;
      result = (_QWORD *)(*v8 & 7LL | v12);
      *v8 = result;
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( v5 == a3 )
        break;
      while ( 1 )
      {
        result = (_QWORD *)sub_34C9970(v5);
        if ( (_BYTE)result )
          break;
        if ( !v5 )
          BUG();
        if ( (*(_BYTE *)v5 & 4) != 0 )
        {
          v5 = *(_QWORD *)(v5 + 8);
          if ( a3 == v5 )
            return result;
        }
        else
        {
          while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
            v5 = *(_QWORD *)(v5 + 8);
          v5 = *(_QWORD *)(v5 + 8);
          if ( a3 == v5 )
            return result;
        }
      }
    }
    while ( a3 != v5 );
  }
  return result;
}
