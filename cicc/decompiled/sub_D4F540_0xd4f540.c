// Function: sub_D4F540
// Address: 0xd4f540
//
void __fastcall sub_D4F540(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  unsigned int v5; // esi
  __int64 v7; // rdi
  int v8; // r11d
  __int64 *v9; // r10
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  int v14; // eax
  int v15; // edx
  unsigned int v16; // esi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // eax
  int v21; // r9d
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v23; // [rsp+18h] [rbp-28h] BYREF

  v22 = a2;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_DWORD *)(a1 + 24);
  if ( a3 )
  {
    if ( v5 )
    {
      v7 = v22;
      v8 = 1;
      v9 = 0;
      v10 = (v5 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v11 = (__int64 *)(v4 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == v22 )
      {
LABEL_4:
        v13 = v11 + 1;
LABEL_5:
        *v13 = a3;
        return;
      }
      while ( v12 != -4096 )
      {
        if ( v12 == -8192 && !v9 )
          v9 = v11;
        v10 = (v5 - 1) & (v8 + v10);
        v11 = (__int64 *)(v4 + 16LL * v10);
        v12 = *v11;
        if ( v22 == *v11 )
          goto LABEL_4;
        ++v8;
      }
      if ( !v9 )
        v9 = v11;
      v14 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v15 = v14 + 1;
      v23 = v9;
      if ( 4 * (v14 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(a1 + 20) - v15 > v5 >> 3 )
        {
LABEL_17:
          *(_DWORD *)(a1 + 16) = v15;
          if ( *v9 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v9 = v7;
          v13 = v9 + 1;
          v9[1] = 0;
          goto LABEL_5;
        }
LABEL_25:
        sub_D4F150(a1, v5);
        sub_D4C730(a1, &v22, &v23);
        v7 = v22;
        v9 = v23;
        v15 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_17;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v23 = 0;
    }
    v5 *= 2;
    goto LABEL_25;
  }
  if ( v5 )
  {
    v16 = v5 - 1;
    v17 = v16 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v18 = (__int64 *)(v4 + 16LL * v17);
    v19 = *v18;
    if ( v22 == *v18 )
    {
LABEL_22:
      *v18 = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v20 = 1;
      while ( v19 != -4096 )
      {
        v21 = v20 + 1;
        v17 = v16 & (v20 + v17);
        v18 = (__int64 *)(v4 + 16LL * v17);
        v19 = *v18;
        if ( v22 == *v18 )
          goto LABEL_22;
        v20 = v21;
      }
    }
  }
}
