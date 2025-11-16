// Function: sub_2AC6E90
// Address: 0x2ac6e90
//
__int64 __fastcall sub_2AC6E90(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v6; // r12
  unsigned int v8; // esi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdi
  int v12; // r11d
  unsigned int v13; // eax
  __int64 v14; // r9
  _QWORD *v15; // rdx
  __int64 v16; // r10
  unsigned int v17; // edx
  __int64 v18; // r12
  __int64 result; // rax
  __int64 v20; // r13
  __int64 v21; // rdx
  int v22; // eax
  int v23; // edi
  unsigned __int64 v24; // r15
  int v25; // ebx
  __int64 v26; // rcx
  __int64 v27; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v28; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 64;
  v27 = a2;
  v8 = *(_DWORD *)(a1 + 88);
  if ( v8 )
  {
    v9 = v27;
    v10 = v8 - 1;
    v11 = *(_QWORD *)(a1 + 72);
    v12 = 1;
    v13 = v10 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v14 = v11 + 56LL * v13;
    v15 = 0;
    v16 = *(_QWORD *)v14;
    if ( v27 == *(_QWORD *)v14 )
    {
LABEL_3:
      v17 = *(_DWORD *)(v14 + 16);
      v18 = v14 + 8;
      result = *a4;
      if ( *((_BYTE *)a4 + 4) != 1 )
      {
LABEL_4:
        v20 = (unsigned int)result;
        if ( (unsigned int)result < v17 )
          goto LABEL_5;
        v24 = (unsigned int)(result + 1);
        result = v17;
        v25 = v24;
        if ( v24 == v17 )
          goto LABEL_5;
        if ( v24 < v17 )
        {
          *(_DWORD *)(v18 + 8) = v24;
          goto LABEL_5;
        }
        goto LABEL_23;
      }
LABEL_8:
      result = (unsigned int)(*(_DWORD *)(a1 + 8) + result);
      goto LABEL_4;
    }
    while ( v16 != -4096 )
    {
      if ( v16 == -8192 && !v15 )
        v15 = (_QWORD *)v14;
      v13 = v10 & (v12 + v13);
      v14 = v11 + 56LL * v13;
      v16 = *(_QWORD *)v14;
      if ( v27 == *(_QWORD *)v14 )
        goto LABEL_3;
      ++v12;
    }
    v22 = *(_DWORD *)(a1 + 80);
    if ( !v15 )
      v15 = (_QWORD *)v14;
    ++*(_QWORD *)(a1 + 64);
    v23 = v22 + 1;
    v28 = v15;
    if ( 4 * (v22 + 1) < 3 * v8 )
    {
      v10 = v8 >> 3;
      if ( v8 - *(_DWORD *)(a1 + 84) - v23 > (unsigned int)v10 )
        goto LABEL_19;
      goto LABEL_33;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 64);
    v28 = 0;
  }
  v8 *= 2;
LABEL_33:
  sub_2AC6C60(v6, v8);
  sub_2ABE350(v6, &v27, &v28);
  v9 = v27;
  v15 = v28;
  v23 = *(_DWORD *)(a1 + 80) + 1;
LABEL_19:
  *(_DWORD *)(a1 + 80) = v23;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v15 = v9;
  v18 = (__int64)(v15 + 1);
  v15[1] = v15 + 3;
  v15[2] = 0x400000000LL;
  LODWORD(result) = *a4;
  if ( *((_BYTE *)a4 + 4) == 1 )
  {
    v17 = 0;
    goto LABEL_8;
  }
  v24 = (unsigned int)(result + 1);
  v20 = (unsigned int)result;
  result = 0;
  v25 = v24;
  if ( !v24 )
  {
LABEL_5:
    v21 = *(_QWORD *)v18;
    goto LABEL_6;
  }
LABEL_23:
  if ( *(unsigned int *)(v18 + 12) < v24 )
  {
    sub_C8D5F0(v18, (const void *)(v18 + 16), v24, 8u, v10, v14);
    result = *(unsigned int *)(v18 + 8);
  }
  v21 = *(_QWORD *)v18;
  result = *(_QWORD *)v18 + 8 * result;
  v26 = *(_QWORD *)v18 + 8 * v24;
  if ( result != v26 )
  {
    do
    {
      if ( result )
        *(_QWORD *)result = 0;
      result += 8;
    }
    while ( v26 != result );
    v21 = *(_QWORD *)v18;
  }
  *(_DWORD *)(v18 + 8) = v25;
LABEL_6:
  *(_QWORD *)(v21 + 8 * v20) = a3;
  return result;
}
