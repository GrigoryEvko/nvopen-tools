// Function: sub_2561130
// Address: 0x2561130
//
__int64 __fastcall sub_2561130(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // r13
  _QWORD *i; // rbx
  int v8; // eax
  _QWORD *v9; // rdi
  int v10; // r10d
  size_t v11; // rdx
  int v12; // r8d
  unsigned int j; // r9d
  __int64 v14; // rax
  const void *v15; // rsi
  unsigned int v16; // r9d
  int v17; // eax
  int v18; // [rsp-5Ch] [rbp-5Ch]
  unsigned int v19; // [rsp-58h] [rbp-58h]
  int v20; // [rsp-54h] [rbp-54h]
  size_t v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-48h] [rbp-48h]
  int v23; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    v5 = (_QWORD *)(result + 16LL * *(unsigned int *)(a1 + 24));
    if ( (_QWORD *)result != v5 )
    {
      while ( 1 )
      {
        v6 = (_QWORD *)result;
        if ( *(_QWORD *)result < 0xFFFFFFFFFFFFFFFELL )
          break;
        result += 16;
        if ( v5 == (_QWORD *)result )
          return result;
      }
      if ( v5 != (_QWORD *)result )
      {
        while ( 1 )
        {
          for ( i = v6 + 2; i != v5; i += 2 )
          {
            if ( *i < 0xFFFFFFFFFFFFFFFELL )
              break;
          }
          v23 = *(_DWORD *)(a2 + 24);
          if ( !v23 )
            goto LABEL_17;
          v22 = *(_QWORD *)(a2 + 8);
          v8 = sub_C94890((_QWORD *)*v6, v6[1]);
          v9 = (_QWORD *)*v6;
          v10 = 1;
          v11 = v6[1];
          v12 = v23 - 1;
          for ( j = (v23 - 1) & v8; ; j = v12 & v16 )
          {
            v14 = v22 + 16LL * j;
            v15 = *(const void **)v14;
            if ( *(_QWORD *)v14 == -1 )
              break;
            if ( v15 == (const void *)-2LL )
            {
              if ( v9 == (_QWORD *)-2LL )
                goto LABEL_18;
            }
            else if ( v11 == *(_QWORD *)(v14 + 8) )
            {
              v18 = v10;
              v19 = j;
              v20 = v12;
              if ( !v11 )
                goto LABEL_18;
              v21 = v11;
              v17 = memcmp(v9, v15, v11);
              v11 = v21;
              v12 = v20;
              j = v19;
              v10 = v18;
              if ( !v17 )
                goto LABEL_18;
            }
            v16 = v10 + j;
            ++v10;
          }
          if ( v9 != (_QWORD *)-1LL )
          {
LABEL_17:
            *v6 = -2;
            v6[1] = 0;
            --*(_DWORD *)(a1 + 16);
            ++*(_DWORD *)(a1 + 20);
          }
LABEL_18:
          result = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 24);
          if ( i == (_QWORD *)result )
            break;
          v6 = i;
        }
      }
    }
  }
  return result;
}
