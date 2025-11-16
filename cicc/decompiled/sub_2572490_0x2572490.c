// Function: sub_2572490
// Address: 0x2572490
//
__int64 __fastcall sub_2572490(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __int64 result; // rax
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r9
  unsigned int v15; // eax
  int v16; // edx
  int v17; // r10d

  v2 = *(_QWORD *)(a1 + 96);
  while ( 1 )
  {
    v3 = *(_QWORD *)(v2 - 32);
    v4 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 != v3 + 48 )
    {
      if ( !v4 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 <= 0xA )
      {
        result = sub_B46E30(v4 - 24);
        v6 = *(_DWORD *)(v2 - 16);
        if ( v6 == (_DWORD)result )
          return result;
        goto LABEL_6;
      }
    }
    v6 = *(_DWORD *)(v2 - 16);
    result = 0;
    if ( !v6 )
      return result;
LABEL_6:
    v7 = *(_QWORD *)(v2 - 24);
    *(_DWORD *)(v2 - 16) = v6 + 1;
    v8 = sub_B46EC0(v7, v6);
    v9 = *(_QWORD *)(a1 + 16);
    v10 = v8;
    v11 = *(unsigned int *)(a1 + 32);
    if ( !(_DWORD)v11 )
      goto LABEL_13;
    v12 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( v10 == *v13 )
    {
LABEL_8:
      if ( v13 == (__int64 *)(v9 + 16 * v11) )
        goto LABEL_13;
      v2 = *(_QWORD *)(a1 + 96);
      v15 = *((_DWORD *)v13 + 2);
      if ( *(_DWORD *)(v2 - 8) > v15 )
      {
        *(_DWORD *)(v2 - 8) = v15;
        v2 = *(_QWORD *)(a1 + 96);
      }
    }
    else
    {
      v16 = 1;
      while ( v14 != -4096 )
      {
        v17 = v16 + 1;
        v12 = (v11 - 1) & (v16 + v12);
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( v10 == *v13 )
          goto LABEL_8;
        v16 = v17;
      }
LABEL_13:
      sub_2572230(a1, v10);
      v2 = *(_QWORD *)(a1 + 96);
    }
  }
}
