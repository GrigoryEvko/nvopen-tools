// Function: sub_D0F960
// Address: 0xd0f960
//
_QWORD *__fastcall sub_D0F960(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *v3; // r13
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  _QWORD *result; // rax
  _QWORD *v10; // rdi

  v2 = a2 + 16;
  v3 = a1 + 2;
  *a1 = *(_QWORD *)a2;
  v5 = *(_QWORD *)(a2 + 24);
  if ( v5 )
  {
    v6 = *(_DWORD *)(a2 + 16);
    a1[3] = v5;
    *((_DWORD *)a1 + 4) = v6;
    a1[4] = *(_QWORD *)(a2 + 32);
    a1[5] = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(v5 + 8) = v3;
    v7 = *(_QWORD *)(a2 + 48);
    *(_QWORD *)(a2 + 24) = 0;
    a1[6] = v7;
    *(_QWORD *)(a2 + 32) = v2;
    *(_QWORD *)(a2 + 40) = v2;
    *(_QWORD *)(a2 + 48) = 0;
  }
  else
  {
    *((_DWORD *)a1 + 4) = 0;
    a1[3] = 0;
    a1[4] = v3;
    a1[5] = v3;
    a1[6] = 0;
  }
  a1[7] = *(_QWORD *)(a2 + 56);
  a1[8] = *(_QWORD *)(a2 + 64);
  v8 = *(_QWORD **)(a2 + 24);
  *(_QWORD *)(a2 + 64) = 0;
  sub_D0EF00(v8);
  result = (_QWORD *)a1[8];
  *(_QWORD *)(a2 + 32) = v2;
  *(_QWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 40) = v2;
  *(_QWORD *)(a2 + 48) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  *result = a1;
  v10 = (_QWORD *)a1[4];
  if ( v3 != v10 )
  {
    do
    {
      *(_QWORD *)v10[5] = a1;
      result = (_QWORD *)sub_220EEE0(v10);
      v10 = result;
    }
    while ( v3 != result );
  }
  return result;
}
