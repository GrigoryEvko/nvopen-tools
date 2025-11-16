// Function: sub_2646530
// Address: 0x2646530
//
__int64 __fastcall sub_2646530(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 **v6; // rbx
  __int64 **v7; // r14
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // r8d
  __int64 v12; // rdi
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // rdi
  unsigned int v16; // edx
  __int64 v17; // rsi
  int v18; // r9d
  __int64 v19; // rdi
  int v20; // edx
  __int64 *v21; // rsi
  __int64 v22; // r8
  int v23; // r8d
  int v24; // esi
  int v25; // r9d
  __int64 v26; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v27; // [rsp+18h] [rbp-98h] BYREF
  char v28[48]; // [rsp+20h] [rbp-90h] BYREF
  char v29[96]; // [rsp+50h] [rbp-60h] BYREF

  v26 = a1;
  sub_26463C0((__int64)v28, a2, &v26);
  result = v26;
  v6 = *(__int64 ***)(v26 + 48);
  v7 = *(__int64 ***)(v26 + 56);
  if ( v6 != v7 )
  {
    while ( 1 )
    {
      v8 = *(_DWORD *)(a2 + 24);
      v9 = *(_QWORD *)(a2 + 8);
      result = **v6;
      v27 = result;
      if ( !v8 )
        goto LABEL_11;
      v10 = v8 - 1;
      v11 = v10 & (((unsigned int)result >> 4) ^ ((unsigned int)result >> 9));
      v12 = *(_QWORD *)(v9 + 8LL * v11);
      if ( result != v12 )
        break;
LABEL_4:
      v13 = *(_DWORD *)(a3 + 24);
      if ( v13 )
      {
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a3 + 8);
        v16 = v14 & (((unsigned int)result >> 4) ^ ((unsigned int)result >> 9));
        v17 = *(_QWORD *)(v15 + 8LL * v16);
        if ( result == v17 )
        {
LABEL_6:
          result = (__int64)*v6;
          *((_BYTE *)*v6 + 17) = 1;
        }
        else
        {
          v23 = 1;
          while ( v17 != -4096 )
          {
            v16 = v14 & (v23 + v16);
            v17 = *(_QWORD *)(v15 + 8LL * v16);
            if ( result == v17 )
              goto LABEL_6;
            ++v23;
          }
        }
      }
LABEL_7:
      v6 += 2;
      if ( v7 == v6 )
        return result;
    }
    v18 = 1;
    while ( v12 != -4096 )
    {
      v11 = v10 & (v18 + v11);
      v12 = *(_QWORD *)(v9 + 8LL * v11);
      if ( result == v12 )
        goto LABEL_4;
      ++v18;
    }
LABEL_11:
    sub_26463C0((__int64)v29, a3, &v27);
    sub_2646530(v27, a2, a3);
    result = *(unsigned int *)(a3 + 24);
    v19 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)result )
    {
      v20 = result - 1;
      result = ((_DWORD)result - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v21 = (__int64 *)(v19 + 8 * result);
      v22 = *v21;
      if ( *v21 == v27 )
      {
LABEL_13:
        *v21 = -8192;
        --*(_DWORD *)(a3 + 16);
        ++*(_DWORD *)(a3 + 20);
      }
      else
      {
        v24 = 1;
        while ( v22 != -4096 )
        {
          v25 = v24 + 1;
          result = v20 & (unsigned int)(v24 + result);
          v21 = (__int64 *)(v19 + 8LL * (unsigned int)result);
          v22 = *v21;
          if ( v27 == *v21 )
            goto LABEL_13;
          v24 = v25;
        }
      }
    }
    goto LABEL_7;
  }
  return result;
}
