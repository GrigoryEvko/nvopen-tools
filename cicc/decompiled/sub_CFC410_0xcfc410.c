// Function: sub_CFC410
// Address: 0xcfc410
//
__int64 __fastcall sub_CFC410(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // rsi
  int v9; // edx
  __int64 v10; // rax
  char *v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // r15
  _QWORD v16[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v17; // [rsp+10h] [rbp-40h]
  int v18; // [rsp+18h] [rbp-38h]

  v17 = a2;
  v6 = *a1;
  v16[0] = 4;
  v16[1] = 0;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v16);
  v18 = -1;
  v7 = *(unsigned int *)(v6 + 8);
  v8 = v7 + 1;
  v9 = v7;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
  {
    v15 = *(_QWORD *)v6;
    if ( *(_QWORD *)v6 > (unsigned __int64)v16 || (unsigned __int64)v16 >= v15 + 32 * v7 )
    {
      sub_CFC2E0(v6, v8, v7, a4, a5, a6);
      v7 = *(unsigned int *)(v6 + 8);
      v10 = *(_QWORD *)v6;
      v11 = (char *)v16;
      v9 = *(_DWORD *)(v6 + 8);
    }
    else
    {
      sub_CFC2E0(v6, v8, v7, a4, a5, a6);
      v10 = *(_QWORD *)v6;
      v7 = *(unsigned int *)(v6 + 8);
      v11 = (char *)v16 + *(_QWORD *)v6 - v15;
      v9 = *(_DWORD *)(v6 + 8);
    }
  }
  else
  {
    v10 = *(_QWORD *)v6;
    v11 = (char *)v16;
  }
  v12 = v10 + 32 * v7;
  if ( v12 )
  {
    *(_QWORD *)v12 = 4;
    v13 = *((_QWORD *)v11 + 2);
    *(_QWORD *)(v12 + 8) = 0;
    *(_QWORD *)(v12 + 16) = v13;
    if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
      sub_BD6050((unsigned __int64 *)v12, *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL);
    *(_DWORD *)(v12 + 24) = *((_DWORD *)v11 + 6);
    v9 = *(_DWORD *)(v6 + 8);
  }
  *(_DWORD *)(v6 + 8) = v9 + 1;
  result = v17;
  if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
    return sub_BD60C0(v16);
  return result;
}
