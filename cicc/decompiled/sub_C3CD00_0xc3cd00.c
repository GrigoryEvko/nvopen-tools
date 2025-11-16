// Function: sub_C3CD00
// Address: 0xc3cd00
//
__int64 __fastcall sub_C3CD00(__int64 a1, __int64 a2)
{
  void **v3; // r13
  void *v5; // rax
  __int64 v6; // rsi
  void *v7; // r14
  __int64 result; // rax
  void **v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rsi
  _QWORD *v12; // rdx
  _QWORD *v13; // rsi
  _QWORD *v14; // rdi
  char v15; // cl
  _QWORD *v16; // rdx
  _QWORD *v17; // rdi
  char v18; // dl

  v3 = *(void ***)(a1 + 8);
  v5 = sub_C33340();
  v6 = *(_QWORD *)(a2 + 8);
  v7 = v5;
  if ( *v3 == v5 )
    result = sub_C3CD00(v3, v6);
  else
    result = sub_C37580((__int64)v3, v6);
  if ( (_DWORD)result == 1 )
  {
    v9 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
    v10 = *(_QWORD *)(a2 + 8) + 24LL;
    result = v7 == *v9 ? sub_C3CD00(v9, v10) : sub_C37580((__int64)v9, v10);
    if ( (result & 0xFFFFFFFD) == 0 )
    {
      v11 = *(_QWORD **)(a1 + 8);
      v12 = v11;
      if ( v7 == (void *)*v11 )
        v12 = (_QWORD *)v11[1];
      if ( v7 == (void *)v11[3] )
        v13 = (_QWORD *)v11[4];
      else
        v13 = v11 + 3;
      v14 = *(_QWORD **)(a2 + 8);
      v15 = ((*((_BYTE *)v13 + 20) & 8) != 0) ^ ((*((_BYTE *)v12 + 20) & 8) != 0);
      v16 = v14;
      if ( v7 == (void *)*v14 )
        v16 = (_QWORD *)v14[1];
      if ( v7 == (void *)v14[3] )
        v17 = (_QWORD *)v14[4];
      else
        v17 = v14 + 3;
      v18 = ((*((_BYTE *)v16 + 20) & 8) != 0) ^ ((*((_BYTE *)v17 + 20) & 8) != 0);
      if ( v18 )
      {
        if ( v15 )
          return (unsigned int)(2 - result);
      }
      else if ( v15 )
      {
        return 0;
      }
      if ( v18 )
        return 2;
    }
  }
  return result;
}
