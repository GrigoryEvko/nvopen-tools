// Function: sub_3554DE0
// Address: 0x3554de0
//
__int64 __fastcall sub_3554DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rcx
  int v13; // edx
  int v14; // r8d
  unsigned int v15; // eax
  __int64 v16; // rdi
  __int64 result; // rax
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rsi
  unsigned __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rcx
  int v24; // edx
  int v25; // r8d
  __int64 v26; // rdi
  __int64 v27; // [rsp+8h] [rbp-48h] BYREF
  __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v27 = a2;
  v28[0] = a2;
  sub_3554C70(a3, v28);
  sub_3554C70(a4, &v27);
  v7 = sub_3545E90(*(_QWORD **)(a1 + 3464), v27);
  v8 = *(__int64 **)v7;
  v9 = *(_QWORD *)v7 + 32LL * *(unsigned int *)(v7 + 8);
  if ( *(_QWORD *)v7 != v9 )
  {
    while ( 1 )
    {
      v10 = *v8;
      if ( ((*((_BYTE *)v8 + 8) ^ 6) & 6) == 0 && *((_DWORD *)v8 + 4) == 3 || *(_DWORD *)(v10 + 200) == -1 )
        goto LABEL_6;
      v11 = *(_DWORD *)(a4 + 24);
      v12 = *(_QWORD *)(a4 + 8);
      if ( !v11 )
        goto LABEL_21;
      v13 = v11 - 1;
      v14 = 1;
      v15 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v16 = *(_QWORD *)(v12 + 8LL * v15);
      if ( v10 != v16 )
        break;
LABEL_6:
      v8 += 4;
      if ( (__int64 *)v9 == v8 )
        goto LABEL_7;
    }
    while ( v16 != -4096 )
    {
      v15 = v13 & (v14 + v15);
      v16 = *(_QWORD *)(v12 + 8LL * v15);
      if ( v10 == v16 )
        goto LABEL_6;
      ++v14;
    }
LABEL_21:
    sub_3554DE0(a1, v10, a3, a4);
    goto LABEL_6;
  }
LABEL_7:
  result = sub_35459D0(*(_QWORD **)(a1 + 3464), v27);
  v18 = *(_QWORD *)result;
  v19 = *(_QWORD *)result + 32LL * *(unsigned int *)(result + 8);
  if ( v19 != *(_QWORD *)result )
  {
    do
    {
      v20 = *(_QWORD *)(v18 + 8);
      result = v20 ^ 6;
      v21 = v20 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (result & 6) != 0 )
      {
        v22 = *(_DWORD *)(a4 + 24);
        v23 = *(_QWORD *)(a4 + 8);
        if ( !v22 )
          goto LABEL_25;
      }
      else
      {
        if ( *(_DWORD *)(v18 + 16) == 3 )
          goto LABEL_11;
        v22 = *(_DWORD *)(a4 + 24);
        v23 = *(_QWORD *)(a4 + 8);
        if ( !v22 )
          goto LABEL_25;
      }
      v24 = v22 - 1;
      v25 = 1;
      result = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v26 = *(_QWORD *)(v23 + 8 * result);
      if ( v21 != v26 )
      {
        while ( v26 != -4096 )
        {
          result = v24 & (unsigned int)(v25 + result);
          v26 = *(_QWORD *)(v23 + 8LL * (unsigned int)result);
          if ( v21 == v26 )
            goto LABEL_11;
          ++v25;
        }
LABEL_25:
        result = sub_3554DE0(a1, v21, a3, a4);
      }
LABEL_11:
      v18 += 32;
    }
    while ( v18 != v19 );
  }
  return result;
}
