// Function: sub_25F8840
// Address: 0x25f8840
//
__int64 __fastcall sub_25F8840(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rdi
  unsigned int v3; // r13d
  __int64 *v5; // rax
  __int64 *v7; // r15
  __int64 v8; // r12
  __int64 *v9; // rbx
  _QWORD *v10; // rdi
  _BYTE *v11; // r11
  int v12; // ecx
  __int64 v13; // rsi
  __int64 v14; // r10
  int v15; // ecx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // eax
  int v24; // ebx
  _BYTE *v26; // [rsp+10h] [rbp-60h] BYREF
  __int64 v27; // [rsp+18h] [rbp-58h]
  _BYTE v28[80]; // [rsp+20h] [rbp-50h] BYREF

  v26 = v28;
  v27 = 0x400000000LL;
  if ( !*(_DWORD *)(a1 + 16) )
    goto LABEL_2;
  v5 = *(__int64 **)(a1 + 8);
  v7 = &v5[2 * *(unsigned int *)(a1 + 24)];
  if ( v5 == v7 )
    goto LABEL_2;
  while ( 1 )
  {
    v8 = *v5;
    v9 = v5;
    if ( *v5 != -4096 && *v5 != -8192 )
      break;
    v5 += 2;
    if ( v7 == v5 )
      goto LABEL_2;
  }
  if ( v7 == v5 )
  {
LABEL_2:
    v2 = v28;
LABEL_3:
    v3 = 1;
    *(_DWORD *)(a2 + 28) = -1;
    goto LABEL_4;
  }
  v10 = (_QWORD *)v5[1];
  v3 = 1;
  if ( (_QWORD *)v10[7] == v10 + 6 )
    goto LABEL_28;
LABEL_13:
  v3 = 0;
  while ( 1 )
  {
    v9 += 2;
    if ( v9 == v7 )
      break;
    while ( *v9 == -4096 || *v9 == -8192 )
    {
      v9 += 2;
      if ( v7 == v9 )
        goto LABEL_18;
    }
    if ( v7 == v9 )
      break;
    v10 = (_QWORD *)v9[1];
    v8 = *v9;
    if ( (_QWORD *)v10[7] != v10 + 6 )
      goto LABEL_13;
LABEL_28:
    sub_AA5450(v10);
    v21 = (unsigned int)v27;
    v22 = (unsigned int)v27 + 1LL;
    if ( v22 > HIDWORD(v27) )
    {
      sub_C8D5F0((__int64)&v26, v28, v22, 8u, v19, v20);
      v21 = (unsigned int)v27;
    }
    *(_QWORD *)&v26[8 * v21] = v8;
    LODWORD(v27) = v27 + 1;
  }
LABEL_18:
  v2 = v26;
  v11 = &v26[8 * (unsigned int)v27];
  if ( v11 != v26 )
  {
    do
    {
      v12 = *(_DWORD *)(a1 + 24);
      v13 = *(_QWORD *)v2;
      v14 = *(_QWORD *)(a1 + 8);
      if ( v12 )
      {
        v15 = v12 - 1;
        v16 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v13 == *v17 )
        {
LABEL_21:
          *v17 = -8192;
          --*(_DWORD *)(a1 + 16);
          ++*(_DWORD *)(a1 + 20);
        }
        else
        {
          v23 = 1;
          while ( v18 != -4096 )
          {
            v24 = v23 + 1;
            v16 = v15 & (v23 + v16);
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( v13 == *v17 )
              goto LABEL_21;
            v23 = v24;
          }
        }
      }
      v2 += 8;
    }
    while ( v2 != v11 );
    v2 = v26;
  }
  if ( (_BYTE)v3 )
    goto LABEL_3;
LABEL_4:
  if ( v2 != v28 )
    _libc_free((unsigned __int64)v2);
  return v3;
}
