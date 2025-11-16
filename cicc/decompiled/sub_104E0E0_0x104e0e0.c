// Function: sub_104E0E0
// Address: 0x104e0e0
//
__int64 __fastcall sub_104E0E0(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  int v6; // edx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 result; // rax
  unsigned int v10; // esi
  __int64 v11; // rdi
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 *v17; // rax
  __int64 v18; // rsi
  char v19; // dl
  __m128i v20; // [rsp+0h] [rbp-50h] BYREF
  char v21; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 104);
  while ( 2 )
  {
    v3 = *(_QWORD *)(v2 - 32);
    v4 = v3 + 48;
    if ( !*(_BYTE *)(v2 - 8) )
    {
      v5 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 == v5 )
      {
        v7 = 0;
      }
      else
      {
        if ( !v5 )
          goto LABEL_24;
        v6 = *(unsigned __int8 *)(v5 - 24);
        v7 = v5 - 24;
        if ( (unsigned int)(v6 - 30) >= 0xB )
          v7 = 0;
      }
      *(_QWORD *)(v2 - 24) = v7;
      *(_DWORD *)(v2 - 16) = 0;
      *(_BYTE *)(v2 - 8) = 1;
    }
LABEL_8:
    v8 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 == v4 )
      goto LABEL_21;
LABEL_9:
    if ( !v8 )
LABEL_24:
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 <= 0xA )
    {
      result = sub_B46E30(v8 - 24);
      v10 = *(_DWORD *)(v2 - 16);
      if ( v10 != (_DWORD)result )
        goto LABEL_12;
LABEL_22:
      *(_QWORD *)(a1 + 104) -= 32LL;
      v2 = *(_QWORD *)(a1 + 104);
      if ( v2 == *(_QWORD *)(a1 + 96) )
        return result;
      continue;
    }
    break;
  }
  while ( 1 )
  {
LABEL_21:
    v10 = *(_DWORD *)(v2 - 16);
    result = 0;
    if ( !v10 )
      goto LABEL_22;
LABEL_12:
    v11 = *(_QWORD *)(v2 - 24);
    *(_DWORD *)(v2 - 16) = v10 + 1;
    v16 = sub_B46EC0(v11, v10);
    if ( *(_BYTE *)(a1 + 28) )
    {
      v17 = *(__int64 **)(a1 + 8);
      v18 = *(unsigned int *)(a1 + 20);
      v12 = &v17[v18];
      if ( v17 != v12 )
      {
        while ( v16 != *v17 )
        {
          if ( v12 == ++v17 )
            goto LABEL_16;
        }
        goto LABEL_8;
      }
LABEL_16:
      if ( (unsigned int)v18 < *(_DWORD *)(a1 + 16) )
        break;
    }
    sub_C8CC70(a1, v16, (__int64)v12, v13, v14, v15);
    if ( v19 )
      goto LABEL_18;
    v8 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 != v4 )
      goto LABEL_9;
  }
  *(_DWORD *)(a1 + 20) = v18 + 1;
  *v12 = v16;
  ++*(_QWORD *)a1;
LABEL_18:
  v20.m128i_i64[0] = v16;
  v21 = 0;
  return sub_104E0A0(a1 + 96, &v20);
}
