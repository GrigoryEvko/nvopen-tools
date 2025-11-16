// Function: sub_2FB03C0
// Address: 0x2fb03c0
//
char __fastcall sub_2FB03C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdi
  int v6; // eax
  int v7; // r8d
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdx
  char result; // al
  int v13; // eax
  __int64 *v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rdi
  _QWORD *v17; // rsi
  int v18; // r10d
  __int64 v19[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)a2;
  v5 = *(_QWORD *)(v3 + 8);
  v6 = *(_DWORD *)(v3 + 24);
  if ( !v6 )
  {
LABEL_10:
    v11 = 0;
    goto LABEL_4;
  }
  v7 = v6 - 1;
  v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( v4 != *v9 )
  {
    v13 = 1;
    while ( v10 != -4096 )
    {
      v18 = v13 + 1;
      v8 = v7 & (v13 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v4 == *v9 )
        goto LABEL_3;
      v13 = v18;
    }
    goto LABEL_10;
  }
LABEL_3:
  v11 = v9[1];
LABEL_4:
  result = *(_BYTE *)(a2 + 32);
  if ( result )
  {
    result = *(_BYTE *)(a2 + 33);
    if ( result )
    {
      result = v11 != 0 && (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0;
      if ( result )
      {
        v19[1] = v2;
        v14 = *(__int64 **)(v11 + 32);
        v19[0] = v4;
        v15 = *v14;
        v16 = *(_QWORD **)(v15 + 64);
        v17 = &v16[*(unsigned int *)(v15 + 72)];
        return v17 != sub_2FB0300(v16, (__int64)v17, v19);
      }
    }
  }
  return result;
}
