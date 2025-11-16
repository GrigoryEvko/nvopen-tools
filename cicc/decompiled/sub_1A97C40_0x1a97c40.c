// Function: sub_1A97C40
// Address: 0x1a97c40
//
__int64 __fastcall sub_1A97C40(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v5; // r13
  __int64 v6; // rbx
  __int64 *v7; // r15
  int v8; // edx
  __int64 v9; // rsi
  __int64 *v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  char *v15; // rsi
  __int64 *v16; // rdx
  __int64 *v17; // rbx
  __int64 v18; // rdx
  char v19; // cl
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r8
  int v23; // r11d
  __int64 *v24; // r10
  unsigned int v25; // edx
  __int64 *v26; // rdi
  __int64 v27; // rcx
  int v28; // edi
  int v29; // ecx
  _BYTE *v30; // rsi
  int v31; // ecx
  int v32; // r8d
  __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  __int64 v35[7]; // [rsp+28h] [rbp-38h] BYREF

  result = (__int64)&v34;
  if ( a2 != a1 )
  {
    v5 = a1;
    while ( 1 )
    {
      v6 = (__int64)(v5 - 3);
      if ( !v5 )
        v6 = 0;
      result = *(unsigned int *)(a3 + 24);
      v35[0] = v6;
      v7 = (__int64 *)v6;
      if ( (_DWORD)result )
      {
        v8 = result - 1;
        v9 = *(_QWORD *)(a3 + 8);
        result = ((_DWORD)result - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v10 = (__int64 *)(v9 + 8 * result);
        v11 = *v10;
        if ( v6 == *v10 )
        {
LABEL_7:
          *v10 = -16;
          v12 = *(_QWORD *)(a3 + 40);
          --*(_DWORD *)(a3 + 16);
          v13 = *(_QWORD **)(a3 + 32);
          ++*(_DWORD *)(a3 + 20);
          result = (__int64)sub_1A94EF0(v13, v12, v35);
          v14 = *(_QWORD *)(a3 + 40);
          v15 = (char *)(result + 8);
          if ( v14 != result + 8 )
          {
            result = (__int64)memmove((void *)result, v15, v14 - (_QWORD)v15);
            v15 = *(char **)(a3 + 40);
          }
          *(_QWORD *)(a3 + 40) = v15 - 8;
        }
        else
        {
          v31 = 1;
          while ( v11 != -8 )
          {
            v32 = v31 + 1;
            result = v8 & (unsigned int)(v31 + result);
            v10 = (__int64 *)(v9 + 8LL * (unsigned int)result);
            v11 = *v10;
            if ( v6 == *v10 )
              goto LABEL_7;
            v31 = v32;
          }
        }
      }
      if ( *(_BYTE *)(v6 + 16) != 77 )
      {
        result = 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v16 = (__int64 *)(v6 - result);
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
        {
          v16 = *(__int64 **)(v6 - 8);
          v7 = (__int64 *)((char *)v16 + result);
        }
        v17 = v16;
        if ( v7 != v16 )
          break;
      }
LABEL_35:
      v5 = (_QWORD *)(*v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( a2 == v5 )
        return result;
    }
    while ( 1 )
    {
      result = *v17;
      v34 = result;
      v18 = *(_QWORD *)result;
      v19 = *(_BYTE *)(*(_QWORD *)result + 8LL);
      if ( v19 == 15 )
      {
        if ( *(_DWORD *)(v18 + 8) >> 8 != 1 )
          goto LABEL_16;
      }
      else
      {
        if ( v19 != 16 )
          goto LABEL_16;
        v20 = *(_QWORD *)(v18 + 24);
        if ( *(_BYTE *)(v20 + 8) != 15 || *(_DWORD *)(v20 + 8) >> 8 != 1 )
          goto LABEL_16;
      }
      if ( *(_BYTE *)(result + 16) <= 0x10u )
        goto LABEL_16;
      v21 = *(_DWORD *)(a3 + 24);
      if ( !v21 )
      {
        ++*(_QWORD *)a3;
        goto LABEL_42;
      }
      v22 = *(_QWORD *)(a3 + 8);
      v23 = 1;
      v24 = 0;
      v25 = (v21 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v26 = (__int64 *)(v22 + 8LL * v25);
      v27 = *v26;
      if ( result == *v26 )
      {
LABEL_16:
        v17 += 3;
        if ( v7 == v17 )
          goto LABEL_35;
      }
      else
      {
        while ( v27 != -8 )
        {
          if ( v27 != -16 || v24 )
            v26 = v24;
          v25 = (v21 - 1) & (v23 + v25);
          v27 = *(_QWORD *)(v22 + 8LL * v25);
          if ( result == v27 )
            goto LABEL_16;
          ++v23;
          v24 = v26;
          v26 = (__int64 *)(v22 + 8LL * v25);
        }
        if ( !v24 )
          v24 = v26;
        v28 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v29 = v28 + 1;
        if ( 4 * (v28 + 1) < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(a3 + 20) - v29 > v21 >> 3 )
            goto LABEL_29;
          goto LABEL_43;
        }
LABEL_42:
        v21 *= 2;
LABEL_43:
        sub_1353F00(a3, v21);
        sub_1A97120(a3, &v34, v35);
        v24 = (__int64 *)v35[0];
        result = v34;
        v29 = *(_DWORD *)(a3 + 16) + 1;
LABEL_29:
        *(_DWORD *)(a3 + 16) = v29;
        if ( *v24 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v24 = result;
        v30 = *(_BYTE **)(a3 + 40);
        if ( v30 == *(_BYTE **)(a3 + 48) )
        {
          result = (__int64)sub_1287830(a3 + 32, v30, &v34);
          goto LABEL_16;
        }
        if ( v30 )
        {
          result = v34;
          *(_QWORD *)v30 = v34;
          v30 = *(_BYTE **)(a3 + 40);
        }
        v17 += 3;
        *(_QWORD *)(a3 + 40) = v30 + 8;
        if ( v7 == v17 )
          goto LABEL_35;
      }
    }
  }
  return result;
}
