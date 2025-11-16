// Function: sub_114A6B0
// Address: 0x114a6b0
//
__int64 __fastcall sub_114A6B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r10
  __int64 result; // rax
  __int64 *v6; // rax
  __int64 *v7; // r9
  __int64 v8; // rdx
  __int64 *v9; // r8
  __int64 v10; // r11
  int v11; // r12d
  _QWORD *v12; // rdi
  __int64 *v13; // rax
  int v14; // eax
  __int64 v15; // rsi
  int v16; // ecx
  unsigned int v17; // eax
  __int64 v18; // rdi
  int v19; // r13d
  __int64 v20; // [rsp-30h] [rbp-30h] BYREF

  v2 = *(unsigned int *)(a1 + 20);
  v3 = *(unsigned int *)(a2 + 40);
  v4 = a2;
  result = 0;
  if ( *(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24) <= (unsigned int)v3 )
  {
    v6 = *(__int64 **)(a1 + 8);
    if ( !*(_BYTE *)(a1 + 28) )
      v2 = *(unsigned int *)(a1 + 16);
    v7 = &v6[v2];
    if ( v6 == v7 )
      return 1;
    while ( 1 )
    {
      v8 = *v6;
      v9 = v6;
      if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v7 == ++v6 )
        return 1;
    }
    if ( v7 == v6 )
    {
      return 1;
    }
    else
    {
      v10 = v3;
      v11 = *(_DWORD *)(a2 + 16);
      v20 = *v6;
      if ( v11 )
        goto LABEL_17;
LABEL_10:
      v12 = *(_QWORD **)(v4 + 32);
      if ( &v12[v10] != sub_1149D50(v12, (__int64)&v12[v10], &v20) )
      {
        do
        {
LABEL_11:
          v13 = v9 + 1;
          if ( v9 + 1 == v7 )
            return 1;
          while ( 1 )
          {
            v8 = *v13;
            v9 = v13;
            if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v7 == ++v13 )
              return 1;
          }
          if ( v7 == v13 )
            return 1;
          v20 = *v13;
          if ( !v11 )
            goto LABEL_10;
LABEL_17:
          v14 = *(_DWORD *)(v4 + 24);
          v15 = *(_QWORD *)(v4 + 8);
          if ( !v14 )
            return 0;
          v16 = v14 - 1;
          v17 = (v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v18 = *(_QWORD *)(v15 + 8LL * v17);
        }
        while ( v18 == v8 );
        v19 = 1;
        while ( v18 != -4096 )
        {
          v17 = v16 & (v19 + v17);
          v18 = *(_QWORD *)(v15 + 8LL * v17);
          if ( v18 == v8 )
            goto LABEL_11;
          ++v19;
        }
      }
      return 0;
    }
  }
  return result;
}
