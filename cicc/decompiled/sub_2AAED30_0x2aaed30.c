// Function: sub_2AAED30
// Address: 0x2aaed30
//
__int64 __fastcall sub_2AAED30(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v6; // r9
  __int64 *v7; // r9
  __int64 v8; // r14
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r8
  _QWORD *v13; // r9
  __int64 result; // rax
  __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = 8LL * a2;
  v6 = *(_QWORD *)(a1 + 8);
  v15[0] = a1;
  v7 = (__int64 *)(v3 + v6);
  v8 = *v7;
  v9 = *(_QWORD **)(*v7 + 16);
  v10 = (__int64)&v9[*(unsigned int *)(*v7 + 24)];
  v11 = sub_2AA89B0(v9, v10, v15);
  if ( (_QWORD *)v10 != v11 )
  {
    if ( (_QWORD *)v10 != v11 + 1 )
    {
      memmove(v11, v11 + 1, v10 - (_QWORD)(v11 + 1));
      LODWORD(v12) = *(_DWORD *)(v8 + 24);
    }
    v12 = (unsigned int)(v12 - 1);
    *(_DWORD *)(v8 + 24) = v12;
    v13 = (_QWORD *)(v3 + *(_QWORD *)(a1 + 8));
  }
  *v13 = a3;
  result = *(unsigned int *)(a3 + 24);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 28) )
  {
    sub_C8D5F0(a3 + 16, (const void *)(a3 + 32), result + 1, 8u, v12, (__int64)v13);
    result = *(unsigned int *)(a3 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * result) = a1;
  ++*(_DWORD *)(a3 + 24);
  return result;
}
