// Function: sub_2C88B20
// Address: 0x2c88b20
//
__int64 __fastcall sub_2C88B20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // edx
  __int64 v5; // rdi
  __int64 v6; // r11
  int v7; // ecx
  __int64 v8; // r10
  __int64 *v9; // rbx
  __int64 v10; // r9
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned int v17; // ebx
  int v18; // ecx
  __int64 v19; // r9
  int v20; // ecx
  unsigned int v21; // r11d
  __int64 *v22; // rdi
  __int64 v23; // r10
  unsigned int v24; // r12d
  int v25; // edi
  int v26; // r12d

  result = *(_QWORD *)(a1 + 48);
  *(_BYTE *)(a1 + 40) = 1;
  v4 = *(_DWORD *)(result + 8);
  if ( v4 )
  {
    v5 = 0;
    v6 = 8LL * (unsigned int)(v4 - 1);
    while ( 1 )
    {
      v7 = *(_DWORD *)(a2 + 24);
      v8 = *(_QWORD *)(a2 + 8);
      v9 = (__int64 *)(v5 + *(_QWORD *)result);
      if ( v7 )
      {
        v10 = *v9;
        v11 = v7 - 1;
        v12 = v11 & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
        result = v8 + 16LL * v12;
        v13 = *(_QWORD *)result;
        if ( v10 == *(_QWORD *)result )
        {
LABEL_6:
          *(_QWORD *)result = -8192;
          --*(_DWORD *)(a2 + 16);
          ++*(_DWORD *)(a2 + 20);
        }
        else
        {
          result = 1;
          while ( v13 != -4096 )
          {
            v24 = result + 1;
            v12 = v11 & (result + v12);
            result = v8 + 16LL * v12;
            v13 = *(_QWORD *)result;
            if ( v10 == *(_QWORD *)result )
              goto LABEL_6;
            result = v24;
          }
        }
      }
      if ( v6 == v5 )
        break;
      result = *(_QWORD *)(a1 + 48);
      v5 += 8;
    }
  }
  v14 = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)v14 )
  {
    v15 = 8 * v14;
    v16 = 0;
    v17 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
    do
    {
      result = *(_QWORD *)(*(_QWORD *)(a1 + 56) + v16);
      if ( result )
      {
        v18 = *(_DWORD *)(result + 32);
        v19 = *(_QWORD *)(result + 16);
        if ( v18 )
        {
          v20 = v18 - 1;
          v21 = v20 & v17;
          v22 = (__int64 *)(v19 + 8LL * (v20 & v17));
          v23 = *v22;
          if ( a1 == *v22 )
          {
LABEL_13:
            *v22 = -8192;
            --*(_DWORD *)(result + 24);
            ++*(_DWORD *)(result + 28);
          }
          else
          {
            v25 = 1;
            while ( v23 != -4096 )
            {
              v26 = v25 + 1;
              v21 = v20 & (v25 + v21);
              v22 = (__int64 *)(v19 + 8LL * v21);
              v23 = *v22;
              if ( a1 == *v22 )
                goto LABEL_13;
              v25 = v26;
            }
          }
        }
        *(_BYTE *)(result + 40) = 1;
        result = *(_QWORD *)(a1 + 56);
        *(_QWORD *)(result + v16) = 0;
      }
      v16 += 8;
    }
    while ( v15 != v16 );
  }
  return result;
}
