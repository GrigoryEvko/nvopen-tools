// Function: sub_D6A020
// Address: 0xd6a020
//
unsigned __int64 __fastcall sub_D6A020(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  _BYTE *v9; // rdi
  _BYTE *v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // eax
  unsigned __int64 result; // rax
  bool v21; // zf
  _BYTE *v22; // [rsp+0h] [rbp-70h] BYREF
  int v23; // [rsp+8h] [rbp-68h]
  _BYTE v24[96]; // [rsp+10h] [rbp-60h] BYREF

  while ( 1 )
  {
    v4 = (_QWORD *)sub_D68C20(**(_QWORD **)a1, a2);
    if ( v4 )
      break;
    v8 = **(_QWORD **)(a1 + 8);
    sub_D69DD0((__int64)&v22, v8, a2, v5, v6, v7);
    v9 = v22;
    v10 = &v22[8 * v23];
    if ( v10 == v22 )
    {
      if ( v10 != v24 )
      {
        v9 = &v22[8 * v23];
LABEL_5:
        _libc_free(v9, v8);
      }
LABEL_6:
      v11 = *(_QWORD *)(a1 + 16);
      if ( a2 )
      {
        v12 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
        v13 = *(_DWORD *)(a2 + 44) + 1;
      }
      else
      {
        v12 = 0;
        v13 = 0;
      }
      if ( v13 >= *(_DWORD *)(v11 + 32) )
        return *(_QWORD *)(**(_QWORD **)a1 + 128LL);
      v14 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 8 * v12);
      if ( !v14 )
        return *(_QWORD *)(**(_QWORD **)a1 + 128LL);
      v15 = *(__int64 **)(v14 + 8);
      if ( !v15 )
        return *(_QWORD *)(**(_QWORD **)a1 + 128LL);
      v16 = *v15;
      if ( a2 == *v15 )
        return *(_QWORD *)(**(_QWORD **)a1 + 128LL);
      goto LABEL_12;
    }
    if ( v10 != v22 + 8 )
    {
      if ( v22 != v24 )
        goto LABEL_5;
      goto LABEL_6;
    }
    v16 = *(_QWORD *)v22;
    if ( v22 != v24 )
      _libc_free(v22, v8);
    v17 = *(_QWORD *)(a1 + 16);
    if ( a2 )
    {
      v18 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
      v19 = *(_DWORD *)(a2 + 44) + 1;
    }
    else
    {
      v18 = 0;
      v19 = 0;
    }
    if ( v19 >= *(_DWORD *)(v17 + 32) || !*(_QWORD *)(*(_QWORD *)(v17 + 24) + 8 * v18) )
      return *(_QWORD *)(**(_QWORD **)a1 + 128LL);
LABEL_12:
    a2 = v16;
  }
  v21 = (*v4 & 0xFFFFFFFFFFFFFFF8LL) == 0;
  result = (*v4 & 0xFFFFFFFFFFFFFFF8LL) - 48;
  if ( v21 )
    return 0;
  return result;
}
