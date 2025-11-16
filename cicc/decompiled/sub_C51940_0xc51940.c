// Function: sub_C51940
// Address: 0xc51940
//
__int64 __fastcall sub_C51940(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int8 a4, char a5)
{
  size_t v5; // rdx
  _BYTE *v6; // r14
  __int64 v7; // r12
  _BYTE *v8; // rax
  signed __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  unsigned int v11; // eax
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __int64 result; // rax
  unsigned int v18; // eax
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v23; // [rsp+10h] [rbp-40h]
  signed __int64 v24; // [rsp+10h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 8);
  if ( !v5 )
    return 0;
  v6 = *(_BYTE **)a2;
  v23 = *(_QWORD *)(a2 + 8);
  v7 = a1 + 128;
  v8 = memchr(*(const void **)a2, 61, v5);
  v9 = v23;
  if ( !v8 || (v10 = v8 - v6, v8 - v6 == -1) )
  {
    v18 = sub_C92610(v6, v23);
    v19 = sub_C92860(v7, v6, v23, v18);
    if ( v19 == -1 )
      return 0;
    v20 = *(_QWORD *)(a1 + 128);
    v21 = v20 + 8LL * v19;
    if ( v21 == v20 + 8LL * *(unsigned int *)(a1 + 136) )
      return 0;
    result = *(_QWORD *)(*(_QWORD *)v21 + 8LL);
  }
  else
  {
    if ( v23 > v10 )
      v9 = v8 - v6;
    v24 = v9;
    v11 = sub_C92610(v6, v9);
    v12 = sub_C92860(v7, v6, v24, v11);
    if ( v12 == -1 )
      return 0;
    v13 = *(_QWORD *)(a1 + 128);
    v14 = v13 + 8LL * v12;
    if ( v14 == v13 + 8LL * *(unsigned int *)(a1 + 136)
      || ((*(_WORD *)(*(_QWORD *)(*(_QWORD *)v14 + 8LL) + 12LL) ^ 0x180) & 0x180) == 0 )
    {
      return 0;
    }
    v15 = *(_QWORD *)(a2 + 8);
    v16 = 0;
    if ( v10 + 1 <= v15 )
    {
      v16 = v15 - (v10 + 1);
      v15 = v10 + 1;
    }
    *a3 = *(_QWORD *)a2 + v15;
    a3[1] = v16;
    if ( *(_QWORD *)(a2 + 8) <= v10 )
      v10 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a2 + 8) = v10;
    result = *(_QWORD *)(*(_QWORD *)v14 + 8LL);
  }
  if ( ((result != 0) & a4) != 0 && !a5 && (*(_BYTE *)(result + 13) & 0x10) == 0 )
    return 0;
  return result;
}
