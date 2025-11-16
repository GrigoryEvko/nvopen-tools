// Function: sub_D696B0
// Address: 0xd696b0
//
unsigned __int64 *__fastcall sub_D696B0(unsigned __int64 *a1, __int64 a2, __int64 *a3)
{
  int v4; // eax
  __int64 v5; // rcx
  int v6; // edx
  __int64 v8; // rsi
  unsigned int v9; // eax
  _QWORD *v10; // r13
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  int v14; // r8d
  _QWORD v15[9]; // [rsp+8h] [rbp-48h] BYREF

  v4 = *(_DWORD *)(a2 + 24);
  if ( !v4 )
    goto LABEL_11;
  v5 = *a3;
  v6 = v4 - 1;
  v8 = *(_QWORD *)(a2 + 8);
  v15[2] = -4096;
  v15[3] = 0;
  v15[0] = 2;
  v15[1] = 0;
  v9 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (_QWORD *)(v8 + ((unsigned __int64)v9 << 6));
  v11 = v10[3];
  if ( v11 != v5 )
  {
    v14 = 1;
    while ( v11 != -4096 )
    {
      v9 = v6 & (v14 + v9);
      v10 = (_QWORD *)(v8 + ((unsigned __int64)v9 << 6));
      v11 = v10[3];
      if ( v5 == v11 )
        goto LABEL_3;
      ++v14;
    }
    sub_D68D70(v15);
    goto LABEL_11;
  }
LABEL_3:
  sub_D68D70(v15);
  if ( v10 == (_QWORD *)(*(_QWORD *)(a2 + 8) + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6)) )
  {
LABEL_11:
    *a1 = 6;
    a1[1] = 0;
    a1[2] = 0;
    return a1;
  }
  *a1 = 6;
  v12 = v10[7];
  a1[1] = 0;
  a1[2] = v12;
  if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
    sub_BD6050(a1, v10[5] & 0xFFFFFFFFFFFFFFF8LL);
  return a1;
}
