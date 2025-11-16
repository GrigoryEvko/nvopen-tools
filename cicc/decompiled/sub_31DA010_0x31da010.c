// Function: sub_31DA010
// Address: 0x31da010
//
__int64 __fastcall sub_31DA010(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  bool v5; // dl
  __int64 result; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // r8
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  int v14; // r10d
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h]

  v15 = 0;
  v16 = 0;
  v17 = a2;
  v5 = a2 != -4096 && a2 != 0 && a2 != -8192;
  if ( v5 )
  {
    sub_BD73F0((__int64)&v15);
    a2 = v17;
    v7 = *(_QWORD *)(a1 + 72);
    v5 = v17 != -4096 && v17 != 0 && v17 != -8192;
    result = *(unsigned int *)(a1 + 88);
    if ( !(_DWORD)result )
      goto LABEL_18;
  }
  else
  {
    result = *(unsigned int *)(a1 + 88);
    v7 = *(_QWORD *)(a1 + 72);
    if ( !(_DWORD)result )
      return result;
  }
  v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (_QWORD *)(v7 + 48LL * v8);
  v10 = v9[2];
  if ( a2 == v10 )
  {
LABEL_4:
    if ( !v5 )
      goto LABEL_5;
    goto LABEL_19;
  }
  v14 = 1;
  while ( v10 != -4096 )
  {
    v8 = (result - 1) & (v14 + v8);
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[2];
    if ( v10 == a2 )
      goto LABEL_4;
    ++v14;
  }
LABEL_18:
  v9 = (_QWORD *)(v7 + 48 * result);
  if ( !v5 )
    return result;
LABEL_19:
  sub_BD60C0(&v15);
  v7 = *(_QWORD *)(a1 + 72);
  result = *(unsigned int *)(a1 + 88);
LABEL_5:
  result = v7 + 48 * result;
  if ( v9 != (_QWORD *)result )
  {
    v11 = *a3;
    v12 = a3[1];
    v13 = a3[2];
    *a3 = v9[3];
    a3[1] = v9[4];
    a3[2] = v9[5];
    v9[3] = v11;
    v9[4] = v12;
    v9[5] = v13;
    if ( v11 )
      j_j___libc_free_0(v11);
    v15 = 0;
    v16 = 0;
    v17 = -8192;
    result = v9[2];
    if ( result != -8192 )
    {
      if ( result != -4096 && result )
        sub_BD60C0(v9);
      v9[2] = -8192;
      result = v17;
      if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        result = sub_BD60C0(&v15);
    }
    --*(_DWORD *)(a1 + 80);
    ++*(_DWORD *)(a1 + 84);
  }
  return result;
}
