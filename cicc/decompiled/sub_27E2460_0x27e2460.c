// Function: sub_27E2460
// Address: 0x27e2460
//
__int64 __fastcall sub_27E2460(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 v10; // r13
  unsigned __int64 v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // r14
  __int64 v14; // [rsp-40h] [rbp-40h]

  v2 = a2[2];
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v5 = *(_QWORD *)(v2 + 24);
    v2 = *(_QWORD *)(v2 + 8);
    if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
      break;
    if ( !v2 )
      return 0;
  }
  while ( v2 )
  {
    v6 = *(_QWORD *)(v2 + 24);
    v2 = *(_QWORD *)(v2 + 8);
    if ( (unsigned __int8)(*(_BYTE *)v6 - 30) <= 0xAu )
    {
      if ( v2 )
      {
        while ( (unsigned __int8)(**(_BYTE **)(v2 + 24) - 30) > 0xAu )
        {
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            goto LABEL_9;
        }
        return 0;
      }
LABEL_9:
      v8 = *(_QWORD *)(v5 + 40);
      v9 = *(_QWORD *)(v6 + 40);
      if ( v9 == v8 )
        return 0;
      v10 = sub_AA54C0(v8);
      if ( !v10 || v10 != sub_AA54C0(v9) )
        return 0;
      v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v11 == v10 + 48 )
        goto LABEL_28;
      if ( !v11 )
        BUG();
      v14 = v11 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_28:
        BUG();
      if ( *(_BYTE *)(v11 - 24) != 31 )
        return 0;
      v12 = (_QWORD *)a2[7];
      if ( a2 + 6 == v12 )
        return 0;
      while ( 1 )
      {
        v13 = 0;
        if ( v12 )
          v13 = (__int64)(v12 - 3);
        if ( sub_D222C0(v13) )
        {
          result = sub_27E1CC0(a1, a2, v13, v14);
          if ( (_BYTE)result )
            break;
        }
        v12 = (_QWORD *)v12[1];
        if ( a2 + 6 == v12 )
          return 0;
      }
      return result;
    }
  }
  return 0;
}
