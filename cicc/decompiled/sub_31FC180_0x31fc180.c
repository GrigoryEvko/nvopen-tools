// Function: sub_31FC180
// Address: 0x31fc180
//
__int64 __fastcall sub_31FC180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  _DWORD *v8; // r13
  char v9; // bl
  char v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rax
  _DWORD *v13; // r15
  _DWORD *v16; // [rsp+18h] [rbp-38h]

  v7 = 0;
  v8 = *(_DWORD **)a2;
  v9 = *(_BYTE *)(a3 + 8);
  v10 = *(_BYTE *)(a2 + 8);
  v16 = *(_DWORD **)a3;
  v11 = *(_QWORD *)a2;
  while ( v10 != v9 )
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
    if ( v16 == (_DWORD *)v11 )
      goto LABEL_10;
    goto LABEL_8;
  }
  if ( v16 != (_DWORD *)v11 )
  {
LABEL_3:
    v11 += 4;
    goto LABEL_4;
  }
LABEL_10:
  v12 = *(unsigned int *)(a1 + 8);
  if ( v7 + v12 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + v12, 4u, a5, a6);
    v12 = *(unsigned int *)(a1 + 8);
    v9 = *(_BYTE *)(a3 + 8);
    v10 = *(_BYTE *)(a2 + 8);
    v8 = *(_DWORD **)a2;
    v16 = *(_DWORD **)a3;
  }
  v13 = (_DWORD *)(*(_QWORD *)a1 + 4 * v12);
  while ( 2 )
  {
    if ( v10 != v9 )
    {
      if ( v10 )
        goto LABEL_21;
LABEL_15:
      if ( v13 )
        *v13 = v8[8];
      v8 = (_DWORD *)sub_220EF30((__int64)v8);
LABEL_18:
      ++v13;
      continue;
    }
    break;
  }
  if ( !v10 )
  {
    if ( v16 == v8 )
      goto LABEL_25;
    goto LABEL_15;
  }
  if ( v16 != v8 )
  {
LABEL_21:
    if ( v13 )
      *v13 = *v8;
    ++v8;
    goto LABEL_18;
  }
LABEL_25:
  *(_DWORD *)(a1 + 8) += v7;
  return a1;
}
