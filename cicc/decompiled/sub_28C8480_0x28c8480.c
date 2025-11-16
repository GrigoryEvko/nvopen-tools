// Function: sub_28C8480
// Address: 0x28c8480
//
__int64 __fastcall sub_28C8480(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // eax
  int v5; // r8d
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r9
  __int64 result; // rax
  int v10; // eax
  int v11; // eax
  __int64 v12; // r8
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  int v17; // r10d
  int v18; // eax
  int v19; // r9d

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(v2 + 40);
  v4 = *(_DWORD *)(v2 + 56);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      result = v7[1];
      if ( result )
        return result;
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v17 = v10 + 1;
        v6 = v5 & (v10 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v17;
      }
    }
  }
  v11 = *(_DWORD *)(a1 + 1784);
  v12 = *(_QWORD *)(a1 + 1768);
  if ( v11 )
  {
    v13 = v11 - 1;
    v14 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( a2 == *v15 )
      return v15[1];
    v18 = 1;
    while ( v16 != -4096 )
    {
      v19 = v18 + 1;
      v14 = v13 & (v18 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        return v15[1];
      v18 = v19;
    }
  }
  return 0;
}
