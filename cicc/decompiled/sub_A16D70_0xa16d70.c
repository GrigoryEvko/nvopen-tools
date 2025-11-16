// Function: sub_A16D70
// Address: 0xa16d70
//
__int64 __fastcall sub_A16D70(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int **v5; // rsi
  __int64 result; // rax
  unsigned int *v7; // r13
  unsigned int *v8; // r15
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r11
  int v17; // edx
  int v18; // r10d
  __int64 v19; // [rsp+0h] [rbp-90h]
  __int64 v20; // [rsp+0h] [rbp-90h]
  unsigned int *v21; // [rsp+10h] [rbp-80h] BYREF
  __int64 v22; // [rsp+18h] [rbp-78h]
  _BYTE v23[112]; // [rsp+20h] [rbp-70h] BYREF

  v5 = &v21;
  v21 = (unsigned int *)v23;
  v22 = 0x400000000LL;
  result = sub_B9A9D0(a3, &v21);
  v7 = v21;
  v8 = &v21[4 * (unsigned int)v22];
  if ( v8 != v21 )
  {
    result = *(unsigned int *)(a2 + 8);
    do
    {
      v9 = *v7;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v20 = *v7;
        sub_C8D5F0(a2, a2 + 16, result + 1, 8);
        result = *(unsigned int *)(a2 + 8);
        v9 = v20;
      }
      v10 = 0xFFFFFFFFLL;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v9;
      v11 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v11;
      v5 = (unsigned int **)*(unsigned int *)(a1 + 304);
      v12 = *((_QWORD *)v7 + 1);
      v13 = *(_QWORD *)(a1 + 288);
      if ( (_DWORD)v5 )
      {
        v5 = (unsigned int **)(unsigned int)((_DWORD)v5 - 1);
        v14 = (unsigned int)v5 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v15 = (__int64 *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( v12 == *v15 )
        {
LABEL_7:
          v10 = (unsigned int)(*((_DWORD *)v15 + 3) - 1);
        }
        else
        {
          v17 = 1;
          while ( v16 != -4096 )
          {
            v18 = v17 + 1;
            v14 = (unsigned int)v5 & (v17 + v14);
            v15 = (__int64 *)(v13 + 16LL * v14);
            v16 = *v15;
            if ( v12 == *v15 )
              goto LABEL_7;
            v17 = v18;
          }
          v10 = 0xFFFFFFFFLL;
        }
      }
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v5 = (unsigned int **)(a2 + 16);
        v19 = v10;
        sub_C8D5F0(a2, a2 + 16, v11 + 1, 8);
        v11 = *(unsigned int *)(a2 + 8);
        v10 = v19;
      }
      v7 += 4;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v10;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    while ( v8 != v7 );
    v7 = v21;
  }
  if ( v7 != (unsigned int *)v23 )
    return _libc_free(v7, v5);
  return result;
}
