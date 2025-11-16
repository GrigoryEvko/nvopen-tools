// Function: sub_FEF430
// Address: 0xfef430
//
__int64 __fastcall sub_FEF430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  _BYTE *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  __int64 result; // rax

  v7 = *(_QWORD *)(a2 + 8);
  if ( !v7 )
    return sub_FEF140(*(_QWORD *)(a1 + 80), *(_DWORD *)(a2 + 16), a3);
  v8 = *(_QWORD *)(**(_QWORD **)(v7 + 32) + 16LL);
  do
  {
    if ( !v8 )
    {
      v13 = 0;
      goto LABEL_10;
    }
    v9 = *(_BYTE **)(v8 + 24);
    v10 = v8;
    v8 = *(_QWORD *)(v8 + 8);
  }
  while ( (unsigned __int8)(*v9 - 30) > 0xAu );
  v11 = v10;
  v12 = 0;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v11 + 8);
    if ( !v11 )
      break;
    while ( (unsigned __int8)(**(_BYTE **)(v11 + 24) - 30) <= 0xAu )
    {
      v11 = *(_QWORD *)(v11 + 8);
      ++v12;
      if ( !v11 )
        goto LABEL_9;
    }
  }
LABEL_9:
  v13 = v12 + 1;
  v8 = v10;
LABEL_10:
  v14 = *(unsigned int *)(a3 + 8);
  if ( v14 + v13 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v14 + v13, 8u, a5, a6);
    v14 = *(unsigned int *)(a3 + 8);
  }
  v15 = (_QWORD *)(*(_QWORD *)a3 + 8 * v14);
  if ( v8 )
  {
    v16 = *(_QWORD *)(v8 + 24);
LABEL_16:
    if ( v15 )
      *v15 = *(_QWORD *)(v16 + 40);
    while ( 1 )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        break;
      v16 = *(_QWORD *)(v8 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v16 - 30) <= 0xAu )
      {
        ++v15;
        goto LABEL_16;
      }
    }
    v14 = *(unsigned int *)(a3 + 8);
  }
  result = v13 + v14;
  *(_DWORD *)(a3 + 8) = result;
  return result;
}
