// Function: sub_30D74B0
// Address: 0x30d74b0
//
__int64 __fastcall sub_30D74B0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // ecx
  __int64 result; // rax
  __int64 v5; // rdi
  int v6; // r8d
  unsigned int v7; // ecx
  __int64 *v8; // rsi
  __int64 v9; // r9
  unsigned int v10; // edx
  int v11; // esi
  int v12; // r10d
  __int64 v13; // [rsp-10h] [rbp-10h]

  v3 = *(_DWORD *)(a2 + 24);
  result = a1;
  v5 = *(_QWORD *)(a2 + 8);
  if ( !v3 )
  {
LABEL_7:
    *(_QWORD *)result = 0;
    *(_DWORD *)(result + 16) = 1;
    *(_QWORD *)(result + 8) = 0;
    return result;
  }
  v6 = v3 - 1;
  v7 = (v3 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v8 = (__int64 *)(v5 + 32LL * v7);
  v9 = *v8;
  if ( a3 != *v8 )
  {
    v11 = 1;
    while ( v9 != -4096 )
    {
      v12 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (__int64 *)(v5 + 32LL * v7);
      v9 = *v8;
      if ( *v8 == a3 )
        goto LABEL_3;
      v11 = v12;
    }
    goto LABEL_7;
  }
LABEL_3:
  *(_QWORD *)result = v8[1];
  v10 = *((_DWORD *)v8 + 6);
  *(_DWORD *)(result + 16) = v10;
  if ( v10 > 0x40 )
  {
    v13 = result;
    sub_C43780(result + 8, (const void **)v8 + 2);
    return v13;
  }
  else
  {
    *(_QWORD *)(result + 8) = v8[2];
  }
  return result;
}
