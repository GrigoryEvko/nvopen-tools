// Function: sub_28D37B0
// Address: 0x28d37b0
//
void __fastcall sub_28D37B0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v6; // rsi
  __int64 v7; // r14
  int v8; // eax
  unsigned int v9; // r8d
  __int64 v10; // rcx
  __int64 v11; // r10
  unsigned int v12; // edx
  _QWORD *v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rsi
  bool v16; // zf
  _QWORD *v17; // rax
  int v18; // ecx
  unsigned int v19; // esi
  int v20; // edx
  __int64 v21; // rdx
  int v22; // ecx
  int v23; // ecx
  __int64 *v24; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+0h] [rbp-80h]
  int v26; // [rsp+8h] [rbp-78h]
  _QWORD *v27; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v28[12]; // [rsp+20h] [rbp-60h] BYREF

  v6 = *(__int64 **)(a2 + 8);
  if ( v6 != a3 && v6 )
    sub_28D33B0(a1, v6, a3);
  v7 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = 0;
  if ( v7 )
  {
    v8 = *(_DWORD *)(v7 + 24);
    if ( !v8 )
    {
      v9 = *(_DWORD *)(a1 + 1880);
      if ( v9 )
      {
        v10 = *(_QWORD *)(v7 + 48);
        v11 = *(_QWORD *)(a1 + 1864);
        v12 = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v13 = (_QWORD *)(v11 + 56LL * v12);
        v14 = *v13;
        if ( v10 == *v13 )
        {
LABEL_8:
          v15 = (__int64)(v13 + 1);
LABEL_9:
          sub_BED950((__int64)v28, v15, (__int64)a3);
          goto LABEL_10;
        }
        v26 = 1;
        v17 = 0;
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v17 )
            v17 = v13;
          v12 = (v9 - 1) & (v26 + v12);
          v13 = (_QWORD *)(v11 + 56LL * v12);
          v14 = *v13;
          if ( v10 == *v13 )
            goto LABEL_8;
          ++v26;
        }
        v22 = *(_DWORD *)(a1 + 1872);
        if ( !v17 )
          v17 = v13;
        ++*(_QWORD *)(a1 + 1856);
        v23 = v22 + 1;
        v28[0] = v17;
        if ( 4 * v23 < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(a1 + 1876) - v23 > v9 >> 3 )
            goto LABEL_27;
          v25 = a1 + 1856;
          sub_28D3590(a1 + 1856, v9);
LABEL_31:
          sub_28CBB90(v25, (__int64 *)(v7 + 48), v28);
          v23 = *(_DWORD *)(a1 + 1872) + 1;
          v17 = (_QWORD *)v28[0];
LABEL_27:
          *(_DWORD *)(a1 + 1872) = v23;
          if ( *v17 == -4096 )
            goto LABEL_18;
          goto LABEL_17;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 1856);
        v28[0] = 0;
      }
      v25 = a1 + 1856;
      sub_28D3590(a1 + 1856, 2 * v9);
      goto LABEL_31;
    }
    if ( v8 != 1 )
      goto LABEL_10;
    v16 = (unsigned __int8)sub_28CBB90(a1 + 1856, (__int64 *)(v7 + 48), &v27) == 0;
    v17 = v27;
    if ( !v16 )
    {
      v15 = (__int64)(v27 + 1);
      goto LABEL_9;
    }
    v18 = *(_DWORD *)(a1 + 1872);
    v19 = *(_DWORD *)(a1 + 1880);
    v28[0] = v27;
    ++*(_QWORD *)(a1 + 1856);
    v20 = v18 + 1;
    if ( 4 * (v18 + 1) >= 3 * v19 )
    {
      v24 = (__int64 *)(v7 + 48);
      sub_28D3590(a1 + 1856, 2 * v19);
    }
    else
    {
      if ( v19 - *(_DWORD *)(a1 + 1876) - v20 > v19 >> 3 )
        goto LABEL_16;
      v24 = (__int64 *)(v7 + 48);
      sub_28D3590(a1 + 1856, v19);
    }
    sub_28CBB90(a1 + 1856, v24, v28);
    v20 = *(_DWORD *)(a1 + 1872) + 1;
    v17 = (_QWORD *)v28[0];
LABEL_16:
    *(_DWORD *)(a1 + 1872) = v20;
    if ( *v17 == -4096 )
    {
LABEL_18:
      v21 = *(_QWORD *)(v7 + 48);
      v15 = (__int64)(v17 + 1);
      *((_BYTE *)v17 + 36) = 1;
      v17[1] = 0;
      *v17 = v21;
      v17[2] = v17 + 5;
      v17[3] = 2;
      *((_DWORD *)v17 + 8) = 0;
      goto LABEL_9;
    }
LABEL_17:
    --*(_DWORD *)(a1 + 1876);
    goto LABEL_18;
  }
LABEL_10:
  *(_QWORD *)(a2 + 16) = 0;
}
