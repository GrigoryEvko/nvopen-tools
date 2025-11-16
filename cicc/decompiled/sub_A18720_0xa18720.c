// Function: sub_A18720
// Address: 0xa18720
//
__int64 __fastcall sub_A18720(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD *v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rbx
  __int64 v10; // rax
  __int64 result; // rax
  _QWORD *v12; // rbx
  __int64 v13; // r13
  char v14; // r8
  _QWORD *v15; // r14
  int v16; // r15d
  __int64 v17; // rax

  if ( a4 )
  {
    sub_A17CC0(a1, a3, 6);
    if ( !*(_DWORD *)(a1 + 48) )
      goto LABEL_3;
  }
  else if ( !*(_DWORD *)(a1 + 48) )
  {
    goto LABEL_3;
  }
  v15 = *(_QWORD **)(a1 + 24);
  v16 = *(_DWORD *)(a1 + 52);
  v17 = v15[1];
  if ( (unsigned __int64)(v17 + 4) > v15[2] )
  {
    sub_C8D290(*(_QWORD *)(a1 + 24), v15 + 3, v17 + 4, 1);
    v17 = v15[1];
  }
  *(_DWORD *)(*v15 + v17) = v16;
  v15[1] += 4LL;
  *(_QWORD *)(a1 + 48) = 0;
LABEL_3:
  v5 = *(_QWORD **)(a1 + 24);
  v6 = v5[1];
  if ( (unsigned __int64)(a3 + v6) > v5[2] )
  {
    sub_C8D290(*(_QWORD *)(a1 + 24), v5 + 3, a3 + v6, 1);
    v6 = v5[1];
  }
  v7 = v6 + *v5;
  if ( a3 > 0 )
  {
    v8 = 0;
    do
    {
      *(_BYTE *)(v7 + v8) = *(_BYTE *)(a2 + v8);
      ++v8;
    }
    while ( a3 != v8 );
    v6 = v5[1];
  }
  v5[1] = v6 + a3;
  while ( 1 )
  {
    result = *(_QWORD *)(a1 + 24);
    v12 = *(_QWORD **)(a1 + 32);
    v13 = *(_QWORD *)(result + 8);
    if ( v12 )
    {
      result = sub_CB7440(*(_QWORD *)(a1 + 32));
      if ( (_BYTE)result )
      {
        if ( !(unsigned __int8)sub_CB7440(v12) )
          BUG();
        v14 = (*(__int64 (__fastcall **)(_QWORD *))(*v12 + 80LL))(v12);
        result = v12[4] - v12[2];
        LOBYTE(v13) = v14 + result + v13;
      }
    }
    if ( (v13 & 3) == 0 )
      break;
    v9 = *(_QWORD **)(a1 + 24);
    v10 = v9[1];
    if ( (unsigned __int64)(v10 + 1) > v9[2] )
    {
      sub_C8D290(*(_QWORD *)(a1 + 24), v9 + 3, v10 + 1, 1);
      v10 = v9[1];
    }
    *(_BYTE *)(*v9 + v10) = 0;
    ++v9[1];
  }
  return result;
}
