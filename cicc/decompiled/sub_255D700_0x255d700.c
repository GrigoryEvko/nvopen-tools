// Function: sub_255D700
// Address: 0x255d700
//
bool __fastcall sub_255D700(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  char v5; // r13
  __int64 v6; // r14
  _QWORD *v7; // rbx
  __int64 *v8; // rcx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  _QWORD *v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rdx

  if ( *(_QWORD *)(a1 + 88) )
  {
    v6 = a1 + 56;
    v5 = 0;
    v4 = *(_QWORD *)(a1 + 72);
  }
  else
  {
    v4 = *(_QWORD *)a1;
    v5 = 1;
    v6 = v4 + 8LL * *(unsigned int *)(a1 + 8);
  }
  v7 = (_QWORD *)(a2 + 56);
  if ( v5 )
    goto LABEL_13;
LABEL_4:
  if ( v4 != v6 )
  {
    v8 = (__int64 *)(v4 + 32);
    if ( *(_QWORD *)(a2 + 88) )
    {
      while ( 1 )
      {
        v12 = *(_QWORD **)(a2 + 64);
        if ( !v12 )
          break;
        v13 = *v8;
        v14 = (_QWORD *)(a2 + 56);
        do
        {
          while ( 1 )
          {
            v15 = v12[2];
            v16 = v12[3];
            if ( v12[4] >= v13 )
              break;
            v12 = (_QWORD *)v12[3];
            if ( !v16 )
              goto LABEL_20;
          }
          v14 = v12;
          v12 = (_QWORD *)v12[2];
        }
        while ( v15 );
LABEL_20:
        if ( v7 == v14 || v13 < v14[4] )
          break;
LABEL_11:
        if ( !v5 )
        {
          v4 = sub_220EF30(v4);
          goto LABEL_4;
        }
        v4 += 8;
LABEL_13:
        if ( v4 == v6 )
          return v4 == v6;
        v8 = (__int64 *)v4;
        if ( !*(_QWORD *)(a2 + 88) )
          goto LABEL_6;
      }
    }
    else
    {
LABEL_6:
      v9 = *(_QWORD **)a2;
      v10 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v10 )
      {
        v11 = *v8;
        while ( *v9 != v11 )
        {
          if ( (_QWORD *)v10 == ++v9 )
            return v4 == v6;
        }
        if ( (_QWORD *)v10 != v9 )
          goto LABEL_11;
      }
    }
  }
  return v4 == v6;
}
