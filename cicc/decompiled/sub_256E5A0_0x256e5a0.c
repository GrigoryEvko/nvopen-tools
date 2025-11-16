// Function: sub_256E5A0
// Address: 0x256e5a0
//
__int64 __fastcall sub_256E5A0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v9; // cl
  int v10; // ecx
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rdi
  __int64 v18; // rbx
  unsigned __int8 *v19; // r12
  unsigned __int8 *v20; // rax
  unsigned int v22; // eax
  int v23; // edx
  unsigned int v24; // edi
  unsigned int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 *v29; // rax
  int v30; // r10d
  __int64 v31; // [rsp+8h] [rbp-38h] BYREF
  __int64 v32; // [rsp+10h] [rbp-30h] BYREF
  int v33; // [rsp+18h] [rbp-28h]

  v9 = *(_BYTE *)(a1 + 584);
  v32 = a2;
  v33 = 0;
  v10 = v9 & 1;
  if ( v10 )
  {
    v11 = a1 + 592;
    v12 = 31;
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 600);
    v11 = *(_QWORD *)(a1 + 592);
    if ( !(_DWORD)v12 )
    {
      v22 = *(_DWORD *)(a1 + 584);
      ++*(_QWORD *)(a1 + 576);
      v31 = 0;
      v23 = (v22 >> 1) + 1;
LABEL_14:
      v24 = 3 * v12;
      goto LABEL_15;
    }
    v12 = (unsigned int)(v12 - 1);
  }
  v13 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = v11 + 16LL * v13;
  a5 = *(_QWORD *)v14;
  if ( a2 == *(_QWORD *)v14 )
  {
LABEL_4:
    v15 = *(unsigned int *)(v14 + 8);
    goto LABEL_5;
  }
  v30 = 1;
  a6 = 0;
  while ( a5 != -4096 )
  {
    if ( !a6 && a5 == -8192 )
      a6 = v14;
    v13 = v12 & (v30 + v13);
    v14 = v11 + 16LL * v13;
    a5 = *(_QWORD *)v14;
    if ( a2 == *(_QWORD *)v14 )
      goto LABEL_4;
    ++v30;
  }
  if ( !a6 )
    a6 = v14;
  v22 = *(_DWORD *)(a1 + 584);
  ++*(_QWORD *)(a1 + 576);
  v31 = a6;
  v23 = (v22 >> 1) + 1;
  if ( !(_BYTE)v10 )
  {
    v12 = *(unsigned int *)(a1 + 600);
    goto LABEL_14;
  }
  v24 = 96;
  v12 = 32;
LABEL_15:
  if ( 4 * v23 >= v24 )
  {
    LODWORD(v12) = 2 * v12;
    goto LABEL_29;
  }
  v25 = v12 - *(_DWORD *)(a1 + 588) - v23;
  v26 = a2;
  if ( v25 <= (unsigned int)v12 >> 3 )
  {
LABEL_29:
    sub_256E180(a1 + 576, v12);
    v12 = (__int64)&v32;
    sub_2566AA0(a1 + 576, &v32, &v31);
    v26 = v32;
    v22 = *(_DWORD *)(a1 + 584);
  }
  v27 = v31;
  *(_DWORD *)(a1 + 584) = (2 * (v22 >> 1) + 2) | v22 & 1;
  if ( *(_QWORD *)v27 != -4096 )
    --*(_DWORD *)(a1 + 588);
  *(_QWORD *)v27 = v26;
  *(_DWORD *)(v27 + 8) = v33;
  v28 = *(unsigned int *)(a1 + 1112);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1116) )
  {
    v12 = a1 + 1120;
    sub_C8D5F0(a1 + 1104, (const void *)(a1 + 1120), v28 + 1, 0x10u, a5, a6);
    v28 = *(unsigned int *)(a1 + 1112);
  }
  v29 = (__int64 *)(*(_QWORD *)(a1 + 1104) + 16 * v28);
  *v29 = a2;
  v29[1] = 0;
  v15 = *(unsigned int *)(a1 + 1112);
  *(_DWORD *)(a1 + 1112) = v15 + 1;
  *(_DWORD *)(v27 + 8) = v15;
LABEL_5:
  v16 = *(_QWORD *)(a1 + 1104) + 16 * v15;
  v17 = *(unsigned __int8 **)(v16 + 8);
  v18 = v16;
  if ( v17 )
  {
    v19 = sub_BD3990(v17, v12);
    if ( v19 == sub_BD3990(a3, v12) )
      return 0;
    v20 = *(unsigned __int8 **)(v18 + 8);
    if ( v20 )
    {
      if ( (unsigned int)*v20 - 12 <= 1 )
        return 0;
    }
  }
  *(_QWORD *)(v18 + 8) = a3;
  return 1;
}
