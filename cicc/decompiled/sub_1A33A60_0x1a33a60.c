// Function: sub_1A33A60
// Address: 0x1a33a60
//
__int64 __fastcall sub_1A33A60(__int64 a1, int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r13
  unsigned __int64 v6; // rdi
  int v7; // r14d
  unsigned int v8; // ebx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v14[4]; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+20h] [rbp-30h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( !a2 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v8 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
    goto LABEL_8;
  }
  v5 = *(__int64 **)(a1 + 16);
  v6 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  v7 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v9 = 32LL * (unsigned int)v6;
    if ( !v4 )
    {
      v8 = *(_DWORD *)(a1 + 24);
      goto LABEL_5;
    }
  }
  else
  {
    if ( !v4 )
    {
      v8 = *(_DWORD *)(a1 + 24);
      v9 = 2048;
      v7 = 64;
LABEL_5:
      v10 = sub_22077B0(v9);
      *(_DWORD *)(a1 + 24) = v7;
      *(_QWORD *)(a1 + 16) = v10;
LABEL_8:
      sub_1A338A0(a1, v5, &v5[4 * v8]);
      return j___libc_free_0(v5);
    }
    v9 = 2048;
    v7 = 64;
  }
  if ( v5 == (__int64 *)-8LL || v5 == (__int64 *)-16LL )
  {
    v12 = v14;
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 24);
    v14[0] = *(_QWORD *)(a1 + 16);
    v12 = (__int64 *)&v15;
    v14[1] = v11;
    v14[2] = *(_QWORD *)(a1 + 32);
    v14[3] = *(_QWORD *)(a1 + 40);
  }
  *(_BYTE *)(a1 + 8) &= ~1u;
  v13 = sub_22077B0(v9);
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v13;
  return sub_1A338A0(a1, v14, v12);
}
