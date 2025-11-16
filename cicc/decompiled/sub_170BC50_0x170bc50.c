// Function: sub_170BC50
// Address: 0x170bc50
//
__int64 __fastcall sub_170BC50(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // r13
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // r9d
  __int64 v11; // rsi
  __int64 v12; // r10
  __int64 *v13; // rcx
  __int64 v14; // rdi

  sub_1AEAA40(a2);
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (unsigned int)v4 <= 7 )
  {
    v5 = 3 * v4;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v7 = *(__int64 **)(a2 - 8);
      v6 = &v7[v5];
    }
    else
    {
      v6 = (__int64 *)a2;
      v7 = (__int64 *)(a2 - v5 * 8);
    }
    for ( ; v6 != v7; v7 += 3 )
    {
      if ( *(_BYTE *)(*v7 + 16) > 0x17u )
        sub_170B990(*(_QWORD *)a1, *v7);
    }
  }
  v8 = *(_QWORD *)a1;
  v9 = *(unsigned int *)(*(_QWORD *)a1 + 2088LL);
  if ( (_DWORD)v9 )
  {
    v10 = 1;
    v11 = *(_QWORD *)(v8 + 2072);
    LODWORD(v12) = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v11 + 16LL * (unsigned int)v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_10:
      if ( v13 != (__int64 *)(v11 + 16 * v9) )
      {
        *(_QWORD *)(*(_QWORD *)v8 + 8LL * *((unsigned int *)v13 + 2)) = 0;
        *v13 = -16;
        --*(_DWORD *)(v8 + 2080);
        ++*(_DWORD *)(v8 + 2084);
      }
    }
    else
    {
      while ( v14 != -8 )
      {
        v12 = ((_DWORD)v9 - 1) & (unsigned int)(v12 + v10);
        v13 = (__int64 *)(v11 + 16 * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_10;
        ++v10;
      }
    }
  }
  sub_15F20C0((_QWORD *)a2);
  *(_BYTE *)(a1 + 2728) = 1;
  return 0;
}
