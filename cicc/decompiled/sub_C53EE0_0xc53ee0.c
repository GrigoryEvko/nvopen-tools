// Function: sub_C53EE0
// Address: 0xc53ee0
//
unsigned __int64 __fastcall sub_C53EE0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  int v10; // r13d
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD v17[4]; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+20h] [rbp-30h]
  char v19; // [rsp+21h] [rbp-2Fh]

  if ( *(_QWORD *)(a1 + 32) )
  {
    if ( *(_QWORD *)(a1 + 136) )
      goto LABEL_3;
  }
  else
  {
    v15 = sub_CEADF0(a1, a2, a3, a4, a5, a6);
    a2 = v17;
    v19 = 1;
    v17[0] = "cl::alias must have argument name specified!";
    v18 = 3;
    sub_C53280(a1, (__int64)v17, 0, 0, v15);
    if ( *(_QWORD *)(a1 + 136) )
      goto LABEL_3;
  }
  v16 = sub_CEADF0(a1, a2, a3, a4, a5, a6);
  a2 = v17;
  v19 = 1;
  v17[0] = "cl::alias must have an cl::aliasopt(option) specified!";
  v18 = 3;
  sub_C53280(a1, (__int64)v17, 0, 0, v16);
LABEL_3:
  if ( *(_DWORD *)(a1 + 116) != *(_DWORD *)(a1 + 120) )
  {
    v6 = sub_CEADF0(a1, a2, a3, a4, a5, a6);
    v19 = 1;
    v17[0] = "cl::alias must not have cl::sub(), aliased option's cl::sub() will be used!";
    v18 = 3;
    sub_C53280(a1, (__int64)v17, 0, 0, v6);
  }
  v7 = *(_QWORD *)(a1 + 136);
  if ( v7 + 96 != a1 + 96 )
  {
    sub_C8CE00(a1 + 96, a1 + 128);
    v7 = *(_QWORD *)(a1 + 136);
  }
  if ( a1 + 72 != v7 + 72 )
  {
    v8 = *(unsigned int *)(v7 + 80);
    v9 = *(unsigned int *)(a1 + 80);
    v10 = *(_DWORD *)(v7 + 80);
    if ( v8 <= v9 )
    {
      if ( *(_DWORD *)(v7 + 80) )
        memmove(*(void **)(a1 + 72), *(const void **)(v7 + 72), 8 * v8);
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 84) )
      {
        v11 = 0;
        *(_DWORD *)(a1 + 80) = 0;
        sub_C8D5F0(a1 + 72, a1 + 88, v8, 8);
        v8 = *(unsigned int *)(v7 + 80);
      }
      else
      {
        v11 = 8 * v9;
        if ( *(_DWORD *)(a1 + 80) )
        {
          memmove(*(void **)(a1 + 72), *(const void **)(v7 + 72), 8 * v9);
          v8 = *(unsigned int *)(v7 + 80);
        }
      }
      v12 = *(_QWORD *)(v7 + 72);
      v13 = 8 * v8;
      if ( v12 + v11 != v13 + v12 )
        memcpy((void *)(v11 + *(_QWORD *)(a1 + 72)), (const void *)(v12 + v11), v13 - v11);
    }
    *(_DWORD *)(a1 + 80) = v10;
  }
  return sub_C53130(a1);
}
