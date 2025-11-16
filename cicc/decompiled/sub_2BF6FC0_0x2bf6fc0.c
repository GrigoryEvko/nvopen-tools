// Function: sub_2BF6FC0
// Address: 0x2bf6fc0
//
__int64 __fastcall sub_2BF6FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rcx
  __int64 v12; // rbx
  _QWORD *v13; // rax
  char v14; // dl
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 *v17; // rdx
  __int64 v18; // rdx
  unsigned int v19; // ecx
  __int64 *v20; // rax
  unsigned __int64 v21; // rdi
  int v22; // ebx
  int v23; // eax
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(_QWORD **)(a1 + 96);
  v8 = *(unsigned int *)(a1 + 104);
  while ( 1 )
  {
    v9 = (__int64)&v7[5 * v8 - 5];
    result = *(_QWORD *)(v9 + 16);
    v11 = *(_QWORD *)(v9 + 24);
    if ( *(_QWORD *)v9 == result && *(_QWORD *)(v9 + 8) == v11 )
      return result;
    *(_QWORD *)(v9 + 24) = v11 + 1;
    if ( !*(_BYTE *)(result + 8) )
    {
      v12 = *(_QWORD *)(result + 112);
      if ( !*(_BYTE *)(a1 + 28) )
        goto LABEL_14;
      goto LABEL_5;
    }
    while ( 1 )
    {
      v9 = *(unsigned int *)(result + 88);
      if ( (_DWORD)v9 )
        break;
      result = *(_QWORD *)(result + 48);
      if ( !result )
        BUG();
    }
    v11 = (unsigned int)v11;
    v12 = *(_QWORD *)(*(_QWORD *)(result + 80) + 8LL * (unsigned int)v11);
    if ( *(_BYTE *)(a1 + 28) )
    {
LABEL_5:
      v13 = *(_QWORD **)(a1 + 8);
      v11 = *(unsigned int *)(a1 + 20);
      v9 = (__int64)&v13[v11];
      if ( v13 == (_QWORD *)v9 )
      {
LABEL_20:
        if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 16) )
          goto LABEL_14;
        v15 = 1;
        *(_DWORD *)(a1 + 20) = v11 + 1;
        *(_QWORD *)v9 = v12;
        LODWORD(v8) = *(_DWORD *)(a1 + 104);
        ++*(_QWORD *)a1;
        if ( !*(_BYTE *)(v12 + 8) )
          goto LABEL_16;
LABEL_22:
        v18 = v12;
        do
        {
          v19 = *(_DWORD *)(v18 + 88);
          if ( v19 )
          {
            v15 = v19;
            goto LABEL_16;
          }
          v18 = *(_QWORD *)(v18 + 48);
        }
        while ( v18 );
        v15 = 0;
        v16 = (unsigned int)v8;
        if ( *(_DWORD *)(a1 + 108) > (unsigned int)v8 )
          goto LABEL_17;
LABEL_26:
        v7 = (_QWORD *)sub_C8D7D0(a1 + 96, a1 + 112, 0, 0x28u, v24, a6);
        v20 = &v7[5 * *(unsigned int *)(a1 + 104)];
        if ( v20 )
        {
          *v20 = v12;
          v20[1] = v15;
          v20[2] = v12;
          v20[3] = 0;
          v20[4] = v12;
        }
        sub_2BF6E30(a1 + 96, v7);
        v21 = *(_QWORD *)(a1 + 96);
        v22 = v24[0];
        if ( a1 + 112 != v21 )
          _libc_free(v21);
        v23 = *(_DWORD *)(a1 + 104);
        *(_QWORD *)(a1 + 96) = v7;
        *(_DWORD *)(a1 + 108) = v22;
        v8 = (unsigned int)(v23 + 1);
        *(_DWORD *)(a1 + 104) = v8;
      }
      else
      {
        while ( v12 != *v13 )
        {
          if ( (_QWORD *)v9 == ++v13 )
            goto LABEL_20;
        }
        v8 = *(unsigned int *)(a1 + 104);
LABEL_10:
        v7 = *(_QWORD **)(a1 + 96);
      }
    }
    else
    {
LABEL_14:
      sub_C8CC70(a1, v12, v9, v11, a5, a6);
      v8 = *(unsigned int *)(a1 + 104);
      if ( !v14 )
        goto LABEL_10;
      v15 = 1;
      if ( *(_BYTE *)(v12 + 8) )
        goto LABEL_22;
LABEL_16:
      v16 = (unsigned int)v8;
      if ( *(_DWORD *)(a1 + 108) <= (unsigned int)v8 )
        goto LABEL_26;
LABEL_17:
      v7 = *(_QWORD **)(a1 + 96);
      v17 = &v7[5 * v16];
      if ( v17 )
      {
        *v17 = v12;
        v17[1] = v15;
        v17[2] = v12;
        v17[3] = 0;
        v17[4] = v12;
        LODWORD(v8) = *(_DWORD *)(a1 + 104);
        v7 = *(_QWORD **)(a1 + 96);
      }
      v8 = (unsigned int)(v8 + 1);
      *(_DWORD *)(a1 + 104) = v8;
    }
  }
}
