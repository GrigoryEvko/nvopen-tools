// Function: sub_2FAF7F0
// Address: 0x2faf7f0
//
_QWORD *__fastcall sub_2FAF7F0(_QWORD *a1, _DWORD *a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  _DWORD *v7; // r14
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  bool v10; // cc
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned int v14; // r12d
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rax
  bool v21; // cf
  unsigned __int64 v22; // rbx
  _DWORD *v24; // [rsp+8h] [rbp-38h]

  result = &a2[a3];
  v24 = result;
  if ( a2 != (_DWORD *)result )
  {
    v7 = a2;
    do
    {
      v8 = *(_QWORD *)(a1[17] + 8LL * (unsigned int)*v7);
      if ( a4 )
      {
        v9 = 2 * v8;
        v10 = v8 <= 2 * v8;
        v8 = -1;
        if ( v10 )
          v8 = v9;
      }
      v11 = *(_QWORD *)(a1[1] + 8LL);
      v12 = (unsigned int)(2 * *v7);
      v13 = (unsigned int)(v12 + 1);
      v14 = *(_DWORD *)(v11 + 4 * v12);
      v15 = *(unsigned int *)(v11 + 4 * v13);
      sub_2FAF160((__int64)a1, v14, a3, v13, a5, a6);
      sub_2FAF160((__int64)a1, v15, v16, v17, v18, v19);
      v20 = (_QWORD *)(a1[3] + 112LL * v14);
      if ( __CFADD__(*v20, v8) )
        *v20 = -1;
      else
        *v20 += v8;
      result = (_QWORD *)(a1[3] + 112 * v15);
      v21 = __CFADD__(*result, v8);
      v22 = *result + v8;
      if ( v21 )
        *result = -1;
      else
        *result = v22;
      ++v7;
    }
    while ( v24 != v7 );
  }
  return result;
}
