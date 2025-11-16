// Function: sub_25BD860
// Address: 0x25bd860
//
char __fastcall sub_25BD860(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax
  int v6; // edx
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 *v11; // r9
  int v12; // ecx
  unsigned int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // r8
  int v16; // edi
  int v17; // r10d
  __int64 v18; // [rsp-20h] [rbp-20h] BYREF

  if ( (unsigned __int8)(*a2 - 34) > 0x33u )
    return 0;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, (unsigned int)*a2 - 34) )
    return 0;
  v4 = *a1;
  if ( !(unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 6) && !(unsigned __int8)sub_B49560((__int64)a2, 6) )
    return 0;
  v5 = *((_QWORD *)a2 - 4);
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(v5 + 24) != *((_QWORD *)a2 + 10) )
    {
      v5 = 0;
    }
  }
  v6 = *(_DWORD *)(v4 + 16);
  v18 = v5;
  if ( v6 )
  {
    v9 = *(_QWORD *)(v4 + 8);
    v10 = *(unsigned int *)(v4 + 24);
    v11 = (__int64 *)(v9 + 8 * v10);
    if ( (_DWORD)v10 )
    {
      v12 = v10 - 1;
      v13 = v12 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v14 = (__int64 *)(v9 + 8LL * v13);
      v15 = *v14;
      if ( v5 == *v14 )
        return v11 == v14;
      v16 = 1;
      while ( v15 != -4096 )
      {
        v17 = v16 + 1;
        v13 = v12 & (v16 + v13);
        v14 = (__int64 *)(v9 + 8LL * v13);
        v15 = *v14;
        if ( v5 == *v14 )
          return v11 == v14;
        v16 = v17;
      }
    }
    return 1;
  }
  else
  {
    v7 = *(_QWORD **)(v4 + 32);
    v8 = &v7[*(unsigned int *)(v4 + 40)];
    return v8 == sub_25BD100(v7, (__int64)v8, &v18);
  }
}
