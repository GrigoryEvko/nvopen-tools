// Function: sub_26E9CC0
// Address: 0x26e9cc0
//
__int64 __fastcall sub_26E9CC0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  int v6; // ecx
  int v7; // r12d
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 *v11; // r8
  int v12; // ecx
  unsigned int v13; // edx
  __int64 *v14; // rsi
  __int64 v15; // r9

  while ( 1 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 == a2 + 48 )
        goto LABEL_15;
      if ( !v4 )
        BUG();
      v5 = v4 - 24;
      v6 = *(unsigned __int8 *)(v4 - 24);
      if ( (unsigned int)(v6 - 30) > 0xA )
LABEL_15:
        BUG();
      if ( (_BYTE)v6 != 34 )
        break;
      a2 = *(_QWORD *)(v4 - 120);
    }
    v7 = sub_B46E30(v4 - 24);
    if ( v7 != 1 )
      return v5;
    v8 = sub_B46EC0(v5, 0);
    v9 = *(_QWORD *)(a3 + 8);
    v10 = *(unsigned int *)(a3 + 24);
    v11 = (__int64 *)(v9 + 8 * v10);
    if ( !(_DWORD)v10 )
      return v5;
    v12 = v10 - 1;
    v13 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v14 = (__int64 *)(v9 + 8LL * v13);
    v15 = *v14;
    if ( v8 != *v14 )
    {
      while ( v15 != -4096 )
      {
        v13 = v12 & (v7 + v13);
        v14 = (__int64 *)(v9 + 8LL * v13);
        v15 = *v14;
        if ( v8 == *v14 )
          goto LABEL_9;
        ++v7;
      }
      return v5;
    }
LABEL_9:
    if ( v11 == v14 )
      return v5;
    a2 = sub_B46EC0(v5, 0);
  }
}
