// Function: sub_2414930
// Address: 0x2414930
//
__int64 __fastcall sub_2414930(__int64 *a1, _BYTE *a2)
{
  __int64 v2; // rcx
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r14d
  unsigned int *v8; // r9
  unsigned int v9; // edx
  unsigned int *v10; // rax
  __int64 v11; // r10
  unsigned int *v12; // r13
  __int64 result; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rsi
  int v19; // eax
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rsi
  int v26; // eax
  int v27; // eax
  int v28; // r10d
  unsigned int *v29; // r8
  _BYTE *v30; // [rsp+8h] [rbp-E8h] BYREF
  _BYTE v31[32]; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v32; // [rsp+30h] [rbp-C0h]
  unsigned int *v33[22]; // [rsp+40h] [rbp-B0h] BYREF

  v2 = (__int64)a2;
  v4 = (__int64)a2;
  v30 = a2;
  if ( *a2 <= 0x1Cu && *a2 != 22 )
    return *(_QWORD *)(*a1 + 40);
  v5 = *((_DWORD *)a1 + 58);
  if ( !v5 )
  {
    ++a1[26];
    v33[0] = 0;
LABEL_9:
    sub_FAA400((__int64)(a1 + 26), 2 * v5);
    v14 = *((_DWORD *)a1 + 58);
    if ( v14 )
    {
      v2 = (__int64)v30;
      v15 = v14 - 1;
      v16 = a1[27];
      v17 = (v14 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v8 = (unsigned int *)(v16 + 16LL * v17);
      v18 = *(_QWORD *)v8;
      if ( v30 == *(_BYTE **)v8 )
      {
LABEL_11:
        v19 = *((_DWORD *)a1 + 56);
        v33[0] = v8;
        v20 = v19 + 1;
      }
      else
      {
        v28 = 1;
        v29 = 0;
        while ( v18 != -4096 )
        {
          if ( !v29 && v18 == -8192 )
            v29 = v8;
          v17 = v15 & (v28 + v17);
          v8 = (unsigned int *)(v16 + 16LL * v17);
          v18 = *(_QWORD *)v8;
          if ( v30 == *(_BYTE **)v8 )
            goto LABEL_11;
          ++v28;
        }
        if ( !v29 )
          v29 = v8;
        v20 = *((_DWORD *)a1 + 56) + 1;
        v33[0] = v29;
        v8 = v29;
      }
    }
    else
    {
      v27 = *((_DWORD *)a1 + 56);
      v2 = (__int64)v30;
      v8 = 0;
      v33[0] = 0;
      v20 = v27 + 1;
    }
    goto LABEL_12;
  }
  v6 = a1[27];
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (unsigned int *)(v6 + 16LL * v9);
  v11 = *(_QWORD *)v10;
  if ( v4 == *(_QWORD *)v10 )
  {
LABEL_5:
    v12 = v10 + 2;
    result = *((_QWORD *)v10 + 1);
    if ( result )
      return result;
    goto LABEL_15;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (unsigned int *)(v6 + 16LL * v9);
    v11 = *(_QWORD *)v10;
    if ( v4 == *(_QWORD *)v10 )
      goto LABEL_5;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v26 = *((_DWORD *)a1 + 56);
  ++a1[26];
  v20 = v26 + 1;
  v33[0] = v8;
  if ( 4 * (v26 + 1) >= 3 * v5 )
    goto LABEL_9;
  if ( v5 - *((_DWORD *)a1 + 57) - v20 <= v5 >> 3 )
  {
    sub_FAA400((__int64)(a1 + 26), v5);
    sub_F9D990((__int64)(a1 + 26), (__int64 *)&v30, v33);
    v2 = (__int64)v30;
    v8 = v33[0];
    v20 = *((_DWORD *)a1 + 56) + 1;
  }
LABEL_12:
  *((_DWORD *)a1 + 56) = v20;
  if ( *(_QWORD *)v8 != -4096 )
    --*((_DWORD *)a1 + 57);
  *((_QWORD *)v8 + 1) = 0;
  v12 = v8 + 2;
  *(_QWORD *)v8 = v2;
  v4 = (__int64)v30;
LABEL_15:
  v21 = *a1;
  if ( *(_BYTE *)v4 != 22 )
    goto LABEL_22;
  if ( *((_BYTE *)a1 + 144) )
    return *(_QWORD *)(v21 + 40);
  if ( (unsigned __int64)*(unsigned int *)(v4 + 32) >= *(_QWORD *)(v21 + 928) )
  {
LABEL_22:
    result = *(_QWORD *)(v21 + 40);
    *(_QWORD *)v12 = result;
    return result;
  }
  v22 = *(_QWORD *)(a1[1] + 80);
  if ( !v22 )
    BUG();
  v23 = *(_QWORD *)(v22 + 32);
  if ( v23 )
    v23 -= 24;
  sub_23D0AB0((__int64)v33, v23, 0, 0, 0);
  v24 = sub_240F400(*a1, *(_DWORD *)(v4 + 32), (__int64 *)v33);
  v32 = 257;
  v25 = *(_QWORD *)(*a1 + 24);
  *(_QWORD *)v12 = sub_A82CA0(v33, v25, v24, 0, 0, (__int64)v31);
  sub_F94A20(v33, v25);
  return *(_QWORD *)v12;
}
