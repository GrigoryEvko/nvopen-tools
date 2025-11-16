// Function: sub_2A6E4B0
// Address: 0x2a6e4b0
//
void __fastcall sub_2A6E4B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r9
  unsigned __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // rdi
  unsigned __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rcx
  int v16; // r10d
  __int64 *v17; // r9
  int v18; // eax
  int v19; // edx
  __int64 *v21; // [rsp+18h] [rbp-48h] BYREF
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v23; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v6 = *a1;
  v7 = sub_22077B0(0x258u);
  v9 = v7;
  if ( v7 )
    sub_2A4B140(v7, a2, a3, a4);
  v10 = *(unsigned int *)(v6 + 2560);
  v22 = v5;
  v23 = v9;
  if ( !(_DWORD)v10 )
  {
    ++*(_QWORD *)(v6 + 2536);
    v21 = 0;
LABEL_18:
    LODWORD(v10) = 2 * v10;
    goto LABEL_19;
  }
  v11 = (unsigned int)(v10 - 1);
  v12 = *(_QWORD *)(v6 + 2544);
  v13 = (unsigned int)v11 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v14 = (__int64 *)(v12 + 16 * v13);
  v15 = *v14;
  if ( v5 == *v14 )
  {
LABEL_5:
    if ( v9 )
    {
      sub_2A45460(v9, v10, (__int64 *)v13, v15, v11, v8);
      j_j___libc_free_0(v9);
    }
    return;
  }
  v16 = 1;
  v17 = 0;
  while ( v15 != -4096 )
  {
    if ( v17 || v15 != -8192 )
      v14 = v17;
    v8 = (unsigned int)(v16 + 1);
    v13 = (unsigned int)v11 & (v16 + (_DWORD)v13);
    v15 = *(_QWORD *)(v12 + 16LL * (unsigned int)v13);
    if ( v5 == v15 )
      goto LABEL_5;
    ++v16;
    v17 = v14;
    v14 = (__int64 *)(v12 + 16LL * (unsigned int)v13);
  }
  if ( !v17 )
    v17 = v14;
  v18 = *(_DWORD *)(v6 + 2552);
  ++*(_QWORD *)(v6 + 2536);
  v19 = v18 + 1;
  v21 = v17;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v10) )
    goto LABEL_18;
  if ( (int)v10 - *(_DWORD *)(v6 + 2556) - v19 <= (unsigned int)v10 >> 3 )
  {
LABEL_19:
    sub_2A6E2A0(v6 + 2536, v10);
    sub_2A66620(v6 + 2536, &v22, &v21);
    v5 = v22;
    v17 = v21;
    v19 = *(_DWORD *)(v6 + 2552) + 1;
  }
  *(_DWORD *)(v6 + 2552) = v19;
  if ( *v17 != -4096 )
    --*(_DWORD *)(v6 + 2556);
  *v17 = v5;
  v17[1] = v23;
}
