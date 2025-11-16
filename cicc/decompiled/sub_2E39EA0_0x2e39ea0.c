// Function: sub_2E39EA0
// Address: 0x2e39ea0
//
__int64 __fastcall sub_2E39EA0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdi
  int v4; // eax
  __int64 v5; // r8
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  int v11; // eax
  int v12; // r10d
  unsigned int v13; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v14; // [rsp-8h] [rbp-8h]

  v3 = *a1;
  if ( !v3 )
    return 0;
  v14 = v2;
  v4 = *(_DWORD *)(v3 + 184);
  v5 = *(_QWORD *)(v3 + 168);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_4:
      v13 = *((_DWORD *)v8 + 2);
      return sub_FE8720(v3, &v13);
    }
    v11 = 1;
    while ( v9 != -4096 )
    {
      v12 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_4;
      v11 = v12;
    }
  }
  v13 = -1;
  return sub_FE8720(v3, &v13);
}
