// Function: sub_2909780
// Address: 0x2909780
//
__int64 __fastcall sub_2909780(_QWORD *a1, _QWORD *a2, __int64 a3, unsigned __int64 a4)
{
  __int64 result; // rax
  _QWORD *v5; // r13
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // r14
  int v11; // edx
  __int64 *v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rsi
  int v17; // r8d
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 v22; // r9
  __int64 v23; // r8
  int v24; // r11d
  unsigned __int64 *v25; // r10
  unsigned __int64 *v26; // rdi
  int v27; // edx
  int v28; // ecx
  _BYTE *v29; // r8
  unsigned int v30; // r8d
  _BYTE *v31; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v32; // [rsp+0h] [rbp-60h]
  const void *v33; // [rsp+8h] [rbp-58h]
  _BYTE *v35; // [rsp+20h] [rbp-40h] BYREF
  __int64 v36[7]; // [rsp+28h] [rbp-38h] BYREF

  result = a3 + 48;
  v33 = (const void *)(a3 + 48);
  if ( a1 != a2 )
  {
    v5 = a1;
    v7 = a4;
    while ( 1 )
    {
      v8 = (__int64)(v5 - 3);
      v9 = *(_QWORD *)(a3 + 8);
      if ( !v5 )
        v8 = 0;
      result = *(unsigned int *)(a3 + 24);
      v36[0] = v8;
      v10 = v8;
      if ( (_DWORD)result )
      {
        v11 = result - 1;
        result = ((_DWORD)result - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v12 = (__int64 *)(v9 + 8 * result);
        v13 = *v12;
        if ( v8 == *v12 )
        {
LABEL_7:
          *v12 = -8192;
          v14 = *(unsigned int *)(a3 + 40);
          --*(_DWORD *)(a3 + 16);
          v15 = *(_QWORD **)(a3 + 32);
          ++*(_DWORD *)(a3 + 20);
          v16 = (__int64)&v15[v14];
          result = (__int64)sub_28FEBC0(v15, v16, v36);
          if ( result + 8 != v16 )
          {
            result = (__int64)memmove((void *)result, (const void *)(result + 8), v16 - (result + 8));
            v17 = *(_DWORD *)(a3 + 40);
          }
          *(_DWORD *)(a3 + 40) = v17 - 1;
        }
        else
        {
          a4 = 1;
          while ( v13 != -4096 )
          {
            v30 = a4 + 1;
            result = v11 & (unsigned int)(a4 + result);
            v12 = (__int64 *)(v9 + 8LL * (unsigned int)result);
            v13 = *v12;
            if ( v8 == *v12 )
              goto LABEL_7;
            a4 = v30;
          }
        }
      }
      if ( *(_BYTE *)v8 != 84 )
      {
        result = 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
        v18 = v8 - result;
        if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
        {
          v18 = *(_QWORD *)(v8 - 8);
          v10 = v18 + result;
        }
        v19 = v18;
        if ( v18 != v10 )
          break;
      }
LABEL_31:
      v5 = (_QWORD *)(*v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v5 == a2 )
        return result;
    }
    while ( 1 )
    {
      v20 = *(_QWORD *)(*(_QWORD *)v19 + 8LL);
      v35 = *(_BYTE **)v19;
      result = sub_28FF380(v20, v7, v18, a4);
      if ( !(_BYTE)result )
        goto LABEL_15;
      result = (__int64)v35;
      if ( *v35 <= 0x15u )
        goto LABEL_15;
      v21 = *(_DWORD *)(a3 + 24);
      if ( !v21 )
      {
        ++*(_QWORD *)a3;
        v36[0] = 0;
        goto LABEL_38;
      }
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a3 + 8);
      v24 = 1;
      v25 = 0;
      v18 = (unsigned int)v22 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v26 = (unsigned __int64 *)(v23 + 8 * v18);
      a4 = *v26;
      if ( v35 == (_BYTE *)*v26 )
      {
LABEL_15:
        v19 += 32;
        if ( v10 == v19 )
          goto LABEL_31;
      }
      else
      {
        while ( a4 != -4096 )
        {
          if ( v25 || a4 != -8192 )
            v26 = v25;
          v18 = (unsigned int)v22 & (v24 + (_DWORD)v18);
          v32 = (unsigned __int64 *)(v23 + 8LL * (unsigned int)v18);
          a4 = *v32;
          if ( v35 == (_BYTE *)*v32 )
            goto LABEL_15;
          ++v24;
          v25 = v26;
          v26 = (unsigned __int64 *)(v23 + 8LL * (unsigned int)v18);
        }
        v27 = *(_DWORD *)(a3 + 16);
        if ( !v25 )
          v25 = v26;
        ++*(_QWORD *)a3;
        v28 = v27 + 1;
        v36[0] = (__int64)v25;
        if ( 4 * (v27 + 1) < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(a3 + 20) - v28 <= v21 >> 3 )
          {
            sub_CE2A30(a3, v21);
            sub_DA5B20(a3, (__int64 *)&v35, v36);
            result = (__int64)v35;
            v25 = (unsigned __int64 *)v36[0];
            v28 = *(_DWORD *)(a3 + 16) + 1;
          }
          goto LABEL_26;
        }
LABEL_38:
        sub_CE2A30(a3, 2 * v21);
        sub_DA5B20(a3, (__int64 *)&v35, v36);
        result = (__int64)v35;
        v25 = (unsigned __int64 *)v36[0];
        v28 = *(_DWORD *)(a3 + 16) + 1;
LABEL_26:
        *(_DWORD *)(a3 + 16) = v28;
        if ( *v25 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v25 = result;
        result = *(unsigned int *)(a3 + 40);
        a4 = *(unsigned int *)(a3 + 44);
        v29 = v35;
        if ( result + 1 > a4 )
        {
          v31 = v35;
          sub_C8D5F0(a3 + 32, v33, result + 1, 8u, (__int64)v35, v22);
          result = *(unsigned int *)(a3 + 40);
          v29 = v31;
        }
        v18 = *(_QWORD *)(a3 + 32);
        v19 += 32;
        *(_QWORD *)(v18 + 8 * result) = v29;
        ++*(_DWORD *)(a3 + 40);
        if ( v10 == v19 )
          goto LABEL_31;
      }
    }
  }
  return result;
}
