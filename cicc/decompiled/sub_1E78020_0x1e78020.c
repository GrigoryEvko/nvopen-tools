// Function: sub_1E78020
// Address: 0x1e78020
//
_BOOL8 __fastcall sub_1E78020(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  _BOOL8 result; // rax
  int v6; // edi
  __int64 v7; // r8
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  _QWORD **v12; // rax
  _QWORD *v13; // rax
  unsigned int i; // edx
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // r9
  _QWORD **v18; // rax
  _QWORD *v19; // rax
  unsigned int v20; // ecx
  int v21; // eax
  int v22; // eax
  int v23; // r10d
  int v24; // r10d

  v4 = *(_QWORD *)(*(_QWORD *)a1 + 272LL);
  result = 0;
  v6 = *(_DWORD *)(v4 + 256);
  if ( !v6 )
    return result;
  v7 = *(_QWORD *)(v4 + 240);
  v8 = v6 - 1;
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v22 = 1;
    while ( v11 != -8 )
    {
      v24 = v22 + 1;
      v9 = v8 & (v22 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v22 = v24;
    }
    goto LABEL_18;
  }
LABEL_3:
  v12 = (_QWORD **)v10[1];
  if ( !v12 )
  {
LABEL_18:
    i = 0;
    goto LABEL_6;
  }
  v13 = *v12;
  for ( i = 1; v13; ++i )
    v13 = (_QWORD *)*v13;
LABEL_6:
  v15 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v16 = (__int64 *)(v7 + 16LL * v15);
  v17 = *v16;
  if ( a3 != *v16 )
  {
    v21 = 1;
    while ( v17 != -8 )
    {
      v23 = v21 + 1;
      v15 = v8 & (v21 + v15);
      v16 = (__int64 *)(v7 + 16LL * v15);
      v17 = *v16;
      if ( a3 == *v16 )
        goto LABEL_7;
      v21 = v23;
    }
    return 0;
  }
LABEL_7:
  v18 = (_QWORD **)v16[1];
  if ( !v18 )
    return 0;
  v19 = *v18;
  v20 = 1;
  if ( !v19 )
    return i == 0;
  do
  {
    v19 = (_QWORD *)*v19;
    ++v20;
  }
  while ( v19 );
  return i < v20;
}
