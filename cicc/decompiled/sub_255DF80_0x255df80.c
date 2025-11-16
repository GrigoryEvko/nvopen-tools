// Function: sub_255DF80
// Address: 0x255df80
//
__int64 __fastcall sub_255DF80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  char v8; // r13
  __int64 *v9; // r12
  char v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 *v17; // [rsp+18h] [rbp-38h]

  v7 = 0;
  v8 = *(_BYTE *)(a3 + 8);
  v9 = *(__int64 **)a2;
  v10 = *(_BYTE *)(a2 + 8);
  v11 = *(_QWORD *)a2;
  v17 = *(__int64 **)a3;
  while ( v10 != v8 )
  {
    if ( v10 )
      goto LABEL_3;
LABEL_8:
    v11 = sub_220EF30(v11);
LABEL_4:
    ++v7;
  }
  if ( !v10 )
  {
    if ( (__int64 *)v11 == v17 )
      goto LABEL_10;
    goto LABEL_8;
  }
  if ( (__int64 *)v11 != v17 )
  {
LABEL_3:
    v11 += 8;
    goto LABEL_4;
  }
LABEL_10:
  v12 = *(unsigned int *)(a1 + 8);
  if ( v7 + v12 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + v12, 8u, a5, a6);
    v12 = *(unsigned int *)(a1 + 8);
    v8 = *(_BYTE *)(a3 + 8);
    v10 = *(_BYTE *)(a2 + 8);
    v9 = *(__int64 **)a2;
    v17 = *(__int64 **)a3;
  }
  v13 = *(_QWORD *)a1 + 8 * v12 + 8;
  while ( 2 )
  {
    if ( v8 != v10 )
    {
      if ( v10 )
        goto LABEL_19;
LABEL_15:
      *(_QWORD *)(v13 - 8) = v9[4];
      v9 = (__int64 *)sub_220EF30((__int64)v9);
LABEL_16:
      v13 += 8;
      continue;
    }
    break;
  }
  if ( !v8 )
  {
    if ( v17 == v9 )
      goto LABEL_21;
    goto LABEL_15;
  }
  if ( v17 != v9 )
  {
LABEL_19:
    v14 = *v9++;
    *(_QWORD *)(v13 - 8) = v14;
    goto LABEL_16;
  }
LABEL_21:
  *(_DWORD *)(a1 + 8) += v7;
  return a1;
}
