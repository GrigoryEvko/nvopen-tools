// Function: sub_28EF5C0
// Address: 0x28ef5c0
//
__int64 __fastcall sub_28EF5C0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r13
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v10; // r8
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // r10d
  int v16; // eax
  __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v17[0] = 0;
    goto LABEL_3;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(a2 + 16);
  v12 = (v4 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v13 = v10 + 32LL * v12;
  v14 = *(_QWORD *)(v13 + 16);
  if ( v14 != v11 )
  {
    v15 = 1;
    v5 = 0;
    while ( v14 != -4096 )
    {
      if ( !v5 && v14 == -8192 )
        v5 = v13;
      v12 = (v4 - 1) & (v15 + v12);
      v13 = v10 + 32LL * v12;
      v14 = *(_QWORD *)(v13 + 16);
      if ( v11 == v14 )
        return v13 + 24;
      ++v15;
    }
    if ( !v5 )
      v5 = v13;
    v16 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v6 = v16 + 1;
    v17[0] = v5;
    if ( 4 * (v16 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 20) - v6 > v4 >> 3 )
      {
LABEL_5:
        *(_DWORD *)(a1 + 16) = v6;
        if ( *(_QWORD *)(v5 + 16) == -4096 )
        {
          v8 = *(_QWORD *)(a2 + 16);
          if ( v8 != -4096 )
          {
LABEL_10:
            *(_QWORD *)(v5 + 16) = v8;
            if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
              sub_BD73F0(v5);
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v7 = *(_QWORD *)(v5 + 16);
          v8 = *(_QWORD *)(a2 + 16);
          if ( v8 != v7 )
          {
            if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
              sub_BD60C0((_QWORD *)v5);
            goto LABEL_10;
          }
        }
        *(_DWORD *)(v5 + 24) = 0;
        return v5 + 24;
      }
LABEL_4:
      sub_28EF240(a1, v4);
      sub_28EE370(a1, a2, v17);
      v5 = v17[0];
      v6 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_5;
    }
LABEL_3:
    v4 *= 2;
    goto LABEL_4;
  }
  return v13 + 24;
}
