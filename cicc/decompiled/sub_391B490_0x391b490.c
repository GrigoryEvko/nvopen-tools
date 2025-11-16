// Function: sub_391B490
// Address: 0x391b490
//
__int64 __fastcall sub_391B490(__int64 a1, __int64 a2, char *a3, size_t a4)
{
  _QWORD *v7; // r15
  __int64 v8; // r8
  __int64 v9; // rax
  size_t v10; // r15
  __int64 v11; // r14
  char v12; // si
  char v13; // al
  char *v14; // rax
  _QWORD *v15; // r15
  void *v16; // rdi
  __int64 result; // rax

  sub_391B370(a1, a2, 0);
  v7 = *(_QWORD **)(a1 + 8);
  v8 = (*(__int64 (__fastcall **)(_QWORD *))(*v7 + 64LL))(v7);
  v9 = v7[3] - v7[1];
  v10 = a4;
  *(_QWORD *)(a2 + 8) = v8 + v9;
  v11 = *(_QWORD *)(a1 + 8);
  do
  {
    while ( 1 )
    {
      v12 = v10 & 0x7F;
      v13 = v10 & 0x7F | 0x80;
      v10 >>= 7;
      if ( v10 )
        v12 = v13;
      v14 = *(char **)(v11 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(v11 + 16) )
        break;
      *(_QWORD *)(v11 + 24) = v14 + 1;
      *v14 = v12;
      if ( !v10 )
        goto LABEL_7;
    }
    sub_16E7DE0(v11, v12);
  }
  while ( v10 );
LABEL_7:
  v15 = *(_QWORD **)(a1 + 8);
  v16 = (void *)v15[3];
  if ( v15[2] - (_QWORD)v16 < a4 )
  {
    sub_16E7EE0(*(_QWORD *)(a1 + 8), a3, a4);
    v15 = *(_QWORD **)(a1 + 8);
  }
  else if ( a4 )
  {
    memcpy(v16, a3, a4);
    v15[3] += a4;
    v15 = *(_QWORD **)(a1 + 8);
  }
  result = (*(__int64 (__fastcall **)(_QWORD *))(*v15 + 64LL))(v15) + v15[3] - v15[1];
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
