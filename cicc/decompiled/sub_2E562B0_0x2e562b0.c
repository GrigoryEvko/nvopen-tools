// Function: sub_2E562B0
// Address: 0x2e562b0
//
void __fastcall sub_2E562B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rdi
  unsigned int v5; // edx
  __int64 *v6; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // r13
  int v9; // r9d

  v3 = *(unsigned int *)(a1 + 104);
  v4 = *(_QWORD *)(a1 + 88);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      goto LABEL_3;
    v9 = 1;
    while ( v7 != -4096 )
    {
      v5 = (v3 - 1) & (v9 + v5);
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_3;
      ++v9;
    }
  }
  v6 = (__int64 *)(v4 + 16 * v3);
LABEL_3:
  v8 = v6[1];
  if ( v8 )
  {
    sub_2E55F30((_QWORD *)v6[1]);
    j_j___libc_free_0(v8);
  }
  *v6 = -8192;
  --*(_DWORD *)(a1 + 96);
  ++*(_DWORD *)(a1 + 100);
}
