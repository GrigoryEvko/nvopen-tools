// Function: sub_2517F50
// Address: 0x2517f50
//
__int64 __fastcall sub_2517F50(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r14
  unsigned int v5; // esi
  unsigned __int64 *v6; // r15
  int v7; // edx
  unsigned __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned __int64 *v12; // r9
  unsigned __int64 v13; // rcx
  int v14; // r10d
  int v15; // eax
  unsigned __int64 *v16; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(__int64 **)(a1 + 32);
  result = 3LL * *(unsigned int *)(a1 + 40);
  v3 = &v1[3 * *(unsigned int *)(a1 + 40)];
  if ( v3 != v1 )
  {
    while ( 1 )
    {
      v5 = *(_DWORD *)(a1 + 24);
      if ( v5 )
      {
        v10 = v1[2];
        v11 = *(_QWORD *)(a1 + 8);
        result = (v5 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v12 = (unsigned __int64 *)(v11 + 24 * result);
        v13 = v12[2];
        if ( v13 == v10 )
          goto LABEL_16;
        v14 = 1;
        v6 = 0;
        while ( v13 != -4096 )
        {
          if ( v6 || v13 != -8192 )
            v12 = v6;
          result = (v5 - 1) & (v14 + (_DWORD)result);
          v13 = *(_QWORD *)(v11 + 24LL * (unsigned int)result + 16);
          if ( v10 == v13 )
            goto LABEL_16;
          ++v14;
          v6 = v12;
          v12 = (unsigned __int64 *)(v11 + 24LL * (unsigned int)result);
        }
        v15 = *(_DWORD *)(a1 + 16);
        if ( !v6 )
          v6 = v12;
        ++*(_QWORD *)a1;
        v7 = v15 + 1;
        v16 = v6;
        if ( 4 * (v15 + 1) < 3 * v5 )
        {
          if ( v5 - *(_DWORD *)(a1 + 20) - v7 > v5 >> 3 )
            goto LABEL_6;
          goto LABEL_5;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
        v16 = 0;
      }
      v5 *= 2;
LABEL_5:
      sub_2517BE0(a1, v5);
      sub_25116B0(a1, (__int64)v1, &v16);
      v6 = v16;
      v7 = *(_DWORD *)(a1 + 16) + 1;
LABEL_6:
      *(_DWORD *)(a1 + 16) = v7;
      if ( v6[2] == -4096 )
      {
        result = v1[2];
        if ( result != -4096 )
          goto LABEL_11;
LABEL_16:
        v1 += 3;
        if ( v3 == v1 )
          return result;
      }
      else
      {
        --*(_DWORD *)(a1 + 20);
        v8 = v6[2];
        result = v1[2];
        if ( result == v8 )
          goto LABEL_16;
        if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
        {
          sub_BD60C0(v6);
          result = v1[2];
        }
LABEL_11:
        v6[2] = result;
        if ( result == 0 || result == -4096 || result == -8192 )
          goto LABEL_16;
        v9 = *v1;
        v1 += 3;
        result = sub_BD6050(v6, v9 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v3 == v1 )
          return result;
      }
    }
  }
  return result;
}
