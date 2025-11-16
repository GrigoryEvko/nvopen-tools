// Function: sub_26E0C10
// Address: 0x26e0c10
//
__int64 __fastcall sub_26E0C10(__int64 a1, __int64 *a2)
{
  int *v3; // r13
  size_t v4; // r12
  int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  int v9; // ecx
  __int64 v10; // r14
  int v11; // edx
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // eax
  __int64 *v15; // rdx
  __int64 v16; // rdi
  int v17; // edx
  int v18; // r8d
  size_t v19[2]; // [rsp+0h] [rbp-E0h] BYREF
  int v20[52]; // [rsp+10h] [rbp-D0h] BYREF

  v3 = (int *)a2[2];
  v4 = a2[3];
  if ( unk_4F838D1 )
  {
    v10 = *a2;
    if ( v3 )
    {
      sub_C7D030(v20);
      sub_C7D280(v20, v3, v4);
      sub_C7D290(v20, v19);
      v4 = v19[0];
    }
    v11 = *(_DWORD *)(v10 + 24);
    v12 = *(_QWORD *)(v10 + 8);
    if ( v11 )
    {
      v13 = v11 - 1;
      v14 = (v11 - 1) & (((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (484763065 * v4));
      v15 = (__int64 *)(v12 + 24LL * v14);
      v16 = *v15;
      if ( v4 == *v15 )
      {
LABEL_20:
        v3 = (int *)v15[1];
        v4 = v15[2];
        goto LABEL_4;
      }
      v17 = 1;
      while ( v16 != -1 )
      {
        v18 = v17 + 1;
        v14 = v13 & (v17 + v14);
        v15 = (__int64 *)(v12 + 24LL * v14);
        v16 = *v15;
        if ( v4 == *v15 )
          goto LABEL_20;
        v17 = v18;
      }
    }
    v4 = 0;
    v3 = 0;
    goto LABEL_4;
  }
  if ( !v3 )
    v4 = 0;
LABEL_4:
  v5 = sub_C92610();
  result = sub_C92860((__int64 *)(a1 + 120), v3, v4, v5);
  if ( (_DWORD)result != -1 )
  {
    v7 = *(_QWORD *)(a1 + 120);
    result = v7 + 8LL * (int)result;
    if ( result != v7 + 8LL * *(unsigned int *)(a1 + 128) )
    {
      result = *(_QWORD *)result;
      if ( *(_QWORD *)(result + 32) )
      {
        result = *(_QWORD *)(result + 24);
        if ( result )
        {
          v8 = *(_QWORD *)(a1 + 360) + 1LL;
          do
          {
            while ( 1 )
            {
              *(_QWORD *)(a1 + 360) = v8;
              v9 = *(_DWORD *)(result + 16);
              if ( (v9 & 0xFFFFFFFB) != 2 && v9 != 4 )
                break;
              ++*(_QWORD *)(a1 + 368);
              result = *(_QWORD *)result;
              ++v8;
              if ( !result )
                return result;
            }
            if ( v9 == 5 )
              ++*(_QWORD *)(a1 + 376);
            result = *(_QWORD *)result;
            ++v8;
          }
          while ( result );
        }
      }
    }
  }
  return result;
}
