// Function: sub_C6AFF0
// Address: 0xc6aff0
//
__int64 __fastcall sub_C6AFF0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v6; // rax
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rdx
  _QWORD *v9; // r12
  __int64 v10; // rcx
  __int64 result; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi

  v6 = (_QWORD *)*a1;
  if ( *a1 )
  {
    LODWORD(v7) = 0;
    do
    {
      v8 = v6;
      v6 = (_QWORD *)*v6;
      v7 = (unsigned int)(v7 + 1);
    }
    while ( v6 );
    v9 = (_QWORD *)v8[1];
    v9[3] = a3;
    v10 = v9[5];
    result = v9[4];
    v9[2] = a2;
    v12 = (v10 - result) >> 4;
    v13 = v12;
    if ( v7 > v12 )
    {
      sub_C6AE30((__int64)(v9 + 4), v7 - v12);
      result = v9[4];
      goto LABEL_9;
    }
  }
  else
  {
    v9 = (_QWORD *)a1[1];
    v10 = v9[5];
    result = v9[4];
    v9[2] = a2;
    v7 = 0;
    v9[3] = a3;
    v13 = (v10 - result) >> 4;
  }
  if ( v7 >= v13 )
    goto LABEL_9;
  v14 = result + 16 * v7;
  if ( v14 == v10 )
    goto LABEL_9;
  v9[5] = v14;
  while ( *a1 )
  {
    result += 16;
    *(_QWORD *)(result - 16) = a1[1];
    *(_DWORD *)(result - 8) = *((_DWORD *)a1 + 4);
    a1 = (_QWORD *)*a1;
LABEL_9:
    ;
  }
  return result;
}
