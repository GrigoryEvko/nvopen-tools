// Function: sub_1F0C4C0
// Address: 0x1f0c4c0
//
__int64 __fastcall sub_1F0C4C0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v7; // r13
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 *v10; // rdi
  int v11; // r10d
  __int64 v12; // r12
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rax
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r14
  __int64 *v20; // rdx
  __int64 *v21; // rcx
  int v23; // eax
  int v24; // edx
  int v25; // r13d
  int v26; // ecx

  if ( a2 != a3 )
  {
    v4 = *(_QWORD *)(a4 + 232);
    v7 = a1;
    v8 = *(_QWORD *)(v4 + 56);
    v9 = *(unsigned int *)(v4 + 72);
    v10 = (__int64 *)(v8 + 16 * v9);
    v11 = v9 - 1;
    while ( 1 )
    {
      v12 = *a2;
      if ( !(_DWORD)v9 )
        return 0;
      v13 = v11 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v14 = (__int64 *)(v8 + 16LL * v13);
      v15 = *v14;
      if ( *v14 != v7 )
        break;
LABEL_5:
      if ( v10 == v14 )
        goto LABEL_23;
      v16 = (__int64 *)v14[1];
LABEL_7:
      v17 = v11 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v18 = (__int64 *)(v8 + 16LL * v17);
      v19 = *v18;
      if ( v12 != *v18 )
      {
        v24 = 1;
        while ( v19 != -8 )
        {
          v25 = v24 + 1;
          v17 = v11 & (v24 + v17);
          v18 = (__int64 *)(v8 + 16LL * v17);
          v19 = *v18;
          if ( v12 == *v18 )
            goto LABEL_8;
          v24 = v25;
        }
        return 0;
      }
LABEL_8:
      if ( v18 == v10 )
        return 0;
      v20 = (__int64 *)v18[1];
      if ( !v16 || !v20 )
        return 0;
      while ( v16 != v20 )
      {
        if ( *((_DWORD *)v16 + 4) < *((_DWORD *)v20 + 4) )
        {
          v21 = v16;
          v16 = v20;
          v20 = v21;
        }
        v16 = (__int64 *)v16[1];
        if ( !v16 )
          return 0;
      }
      v7 = *v16;
      if ( !*v16 )
        return 0;
      if ( a3 == ++a2 )
      {
        if ( a1 == v7 )
          return 0;
        return v7;
      }
    }
    v23 = 1;
    while ( v15 != -8 )
    {
      v26 = v23 + 1;
      v13 = v11 & (v23 + v13);
      v14 = (__int64 *)(v8 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == v7 )
        goto LABEL_5;
      v23 = v26;
    }
LABEL_23:
    v16 = 0;
    goto LABEL_7;
  }
  return 0;
}
