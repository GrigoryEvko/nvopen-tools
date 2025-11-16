// Function: sub_1E62D80
// Address: 0x1e62d80
//
__int64 __fastcall sub_1E62D80(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  __int64 *v2; // r12
  __int64 *v3; // r14
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  int v12; // edx
  int v13; // r9d
  __int64 v15; // [rsp+8h] [rbp-38h]

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(__int64 **)(v1 + 72);
  v3 = *(__int64 **)(v1 + 64);
  if ( v3 == v2 )
    return 0;
  v15 = 0;
  do
  {
    v4 = a1[3];
    v5 = *v3;
    sub_1E06620(v4);
    v6 = *(_QWORD *)(v4 + 1312);
    v7 = *(unsigned int *)(v6 + 48);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD *)(v6 + 32);
      v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v5 == *v10 )
      {
LABEL_3:
        if ( v10 != (__int64 *)(v8 + 16 * v7) && v10[1] && !(unsigned __int8)sub_1E62AD0(a1, v5) )
        {
          if ( v15 )
            return 0;
          v15 = v5;
        }
      }
      else
      {
        v12 = 1;
        while ( v11 != -8 )
        {
          v13 = v12 + 1;
          v9 = (v7 - 1) & (v12 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v5 == *v10 )
            goto LABEL_3;
          v12 = v13;
        }
      }
    }
    ++v3;
  }
  while ( v2 != v3 );
  return v15;
}
