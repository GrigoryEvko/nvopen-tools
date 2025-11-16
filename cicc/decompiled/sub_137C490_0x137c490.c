// Function: sub_137C490
// Address: 0x137c490
//
__int64 __fastcall sub_137C490(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx
  unsigned int v10; // edi
  __int64 *v11; // rdx
  __int64 v12; // r9
  unsigned int v13; // eax
  int v14; // edx
  int v15; // r10d

  v2 = *(_QWORD *)(a1 + 96);
  while ( 1 )
  {
    v3 = sub_157EBA0(*(_QWORD *)(v2 - 32));
    result = 0;
    if ( v3 )
    {
      result = sub_15F4D60(v3);
      v2 = *(_QWORD *)(a1 + 96);
    }
    v5 = *(unsigned int *)(v2 - 16);
    if ( (_DWORD)v5 == (_DWORD)result )
      return result;
    v6 = *(_QWORD *)(v2 - 24);
    *(_DWORD *)(v2 - 16) = v5 + 1;
    v7 = sub_15F4DF0(v6, v5);
    v8 = *(unsigned int *)(a1 + 32);
    if ( !(_DWORD)v8 )
      goto LABEL_12;
    v9 = *(_QWORD *)(a1 + 16);
    v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v7 == *v11 )
    {
LABEL_7:
      if ( v11 == (__int64 *)(v9 + 16 * v8) )
        goto LABEL_12;
      v2 = *(_QWORD *)(a1 + 96);
      v13 = *((_DWORD *)v11 + 2);
      if ( *(_DWORD *)(v2 - 8) > v13 )
      {
        *(_DWORD *)(v2 - 8) = v13;
        v2 = *(_QWORD *)(a1 + 96);
      }
    }
    else
    {
      v14 = 1;
      while ( v12 != -8 )
      {
        v15 = v14 + 1;
        v10 = (v8 - 1) & (v14 + v10);
        v11 = (__int64 *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
          goto LABEL_7;
        v14 = v15;
      }
LABEL_12:
      sub_137C180(a1, v7);
      v2 = *(_QWORD *)(a1 + 96);
    }
  }
}
