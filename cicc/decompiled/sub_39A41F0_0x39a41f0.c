// Function: sub_39A41F0
// Address: 0x39a41f0
//
__int64 __fastcall sub_39A41F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  void *v7; // rax
  size_t v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rcx
  int v12; // r9d
  unsigned int v13; // r10d
  __int64 *v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // r12
  unsigned int v17; // r15d
  unsigned __int64 v18; // r8
  __int64 result; // rax
  __int64 v20; // r8
  __int64 v21[8]; // [rsp+0h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a3 + 8 * (2LL - *(unsigned int *)(a3 + 8)));
  if ( v6 )
  {
    v7 = (void *)sub_161E970(v6);
    if ( v8 )
      sub_39A3F30(a1, a2, 3, v7, v8);
  }
  v9 = a1[25];
  v10 = *(unsigned int *)(v9 + 5440);
  if ( !(_DWORD)v10 )
  {
LABEL_14:
    v16 = (__int64 *)(a2 + 8);
    goto LABEL_9;
  }
  v11 = *(_QWORD *)(v9 + 5424);
  v12 = 1;
  v13 = (v10 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( a3 != *v14 )
  {
    while ( v15 != -8 )
    {
      v13 = (v10 - 1) & (v13 + v12);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( a3 == *v14 )
        goto LABEL_6;
      ++v12;
    }
    goto LABEL_14;
  }
LABEL_6:
  v16 = (__int64 *)(a2 + 8);
  if ( v14 != (__int64 *)(v11 + 16 * v10) )
  {
    v17 = *((_DWORD *)v14 + 2);
    if ( v17 )
    {
      v21[0] = 0x1900000009LL;
      HIWORD(v21[0]) = (unsigned __int16)sub_398C0A0(v9) < 4u ? 6 : 23;
      v21[1] = v17;
      sub_39A31C0(v16, a1 + 11, v21);
    }
  }
LABEL_9:
  v18 = *(_QWORD *)(a3 + 32);
  BYTE2(v21[0]) = 0;
  result = sub_39A3560((__int64)a1, v16, 11, (__int64)v21, v18 >> 3);
  v20 = *(unsigned int *)(a3 + 52);
  if ( (_DWORD)v20 )
  {
    LODWORD(v21[0]) = 65547;
    return sub_39A3560((__int64)a1, v16, 62, (__int64)v21, v20);
  }
  return result;
}
