// Function: sub_A6E600
// Address: 0xa6e600
//
__int64 __fastcall sub_A6E600(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  unsigned __int64 v4; // r15
  __int64 result; // rax
  unsigned __int64 v7; // r13
  unsigned int v8; // edx
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  unsigned int v11; // eax
  unsigned __int64 v12; // r15
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  unsigned __int64 v19; // rdx
  int v20; // edi
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  int v23; // eax
  unsigned __int64 v24; // r10
  char v25; // [rsp+7h] [rbp-69h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  unsigned __int64 v28; // [rsp+20h] [rbp-50h]
  unsigned __int64 v29; // [rsp+20h] [rbp-50h]
  int v30; // [rsp+20h] [rbp-50h]
  unsigned __int64 v31; // [rsp+28h] [rbp-48h]
  __int64 v32[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = 32 * a3;
  v4 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  result = a2 + v3;
  v27 = a1 + 16;
  *(_QWORD *)a1 = a1 + 16;
  v31 = a2 + v3;
  if ( a2 != a2 + v3 )
  {
    v7 = a2;
    v8 = 0;
    v9 = a2 + 32;
    v10 = 0;
    while ( 1 )
    {
      result = 32 * v10;
      v12 = result + v4;
      if ( !v12 )
        goto LABEL_5;
      v13 = *(_DWORD *)(v7 + 8);
      *(_DWORD *)(v12 + 8) = v13;
      if ( v13 <= 0x40 )
      {
        *(_QWORD *)v12 = *(_QWORD *)v7;
        v11 = *(_DWORD *)(v7 + 24);
        *(_DWORD *)(v12 + 24) = v11;
        if ( v11 <= 0x40 )
          goto LABEL_4;
      }
      else
      {
        sub_C43780(v12, v7);
        v14 = *(_DWORD *)(v7 + 24);
        *(_DWORD *)(v12 + 24) = v14;
        if ( v14 <= 0x40 )
        {
LABEL_4:
          result = *(_QWORD *)(v7 + 16);
          *(_QWORD *)(v12 + 16) = result;
          v8 = *(_DWORD *)(a1 + 8);
LABEL_5:
          ++v8;
          v7 = v9;
          *(_DWORD *)(a1 + 8) = v8;
          if ( v9 == v31 )
            return result;
          goto LABEL_6;
        }
      }
      v15 = v7 + 16;
      v7 = v9;
      result = sub_C43780(v12 + 16, v15);
      v8 = *(_DWORD *)(a1 + 8) + 1;
      *(_DWORD *)(a1 + 8) = v8;
      if ( v9 == v31 )
        return result;
LABEL_6:
      v10 = v8;
      v4 = *(_QWORD *)a1;
      if ( (unsigned __int64)v8 + 1 > *(unsigned int *)(a1 + 12) )
      {
        if ( v4 > v9 || v4 + 32LL * v8 <= v9 )
        {
          v26 = -1;
          v25 = 0;
        }
        else
        {
          v25 = 1;
          v26 = (__int64)(v9 - v4) >> 5;
        }
        v4 = sub_C8D7D0(a1, v27, v8 + 1LL, 32, v32);
        v16 = *(_QWORD *)a1;
        v17 = 32LL * *(unsigned int *)(a1 + 8);
        v18 = *(_QWORD *)a1 + v17;
        if ( *(_QWORD *)a1 != v18 )
        {
          v17 += v4;
          v19 = v4;
          do
          {
            if ( v19 )
            {
              *(_DWORD *)(v19 + 8) = *(_DWORD *)(v16 + 8);
              *(_QWORD *)v19 = *(_QWORD *)v16;
              v20 = *(_DWORD *)(v16 + 24);
              *(_DWORD *)(v16 + 8) = 0;
              *(_DWORD *)(v19 + 24) = v20;
              *(_QWORD *)(v19 + 16) = *(_QWORD *)(v16 + 16);
              *(_DWORD *)(v16 + 24) = 0;
            }
            v19 += 32LL;
            v16 += 32LL;
          }
          while ( v19 != v17 );
          v21 = *(_QWORD *)a1;
          v18 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
          if ( *(_QWORD *)a1 != v18 )
          {
            do
            {
              v18 -= 32;
              if ( *(_DWORD *)(v18 + 24) > 0x40u )
              {
                v22 = *(_QWORD *)(v18 + 16);
                if ( v22 )
                {
                  v28 = v21;
                  j_j___libc_free_0_0(v22);
                  v21 = v28;
                }
              }
              if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
              {
                v29 = v21;
                j_j___libc_free_0_0(*(_QWORD *)v18);
                v21 = v29;
              }
            }
            while ( v18 != v21 );
            v18 = *(_QWORD *)a1;
          }
        }
        v23 = v32[0];
        if ( v27 != v18 )
        {
          v30 = v32[0];
          _libc_free(v18, v17);
          v23 = v30;
        }
        v24 = v7;
        *(_DWORD *)(a1 + 12) = v23;
        v10 = *(unsigned int *)(a1 + 8);
        *(_QWORD *)a1 = v4;
        v8 = v10;
        if ( v25 )
          v24 = v4 + 32 * v26;
        v7 = v24;
      }
      v9 += 32LL;
    }
  }
  return result;
}
