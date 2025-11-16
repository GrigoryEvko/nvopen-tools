// Function: sub_38E0570
// Address: 0x38e0570
//
__int64 __fastcall sub_38E0570(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 result; // rax
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  unsigned __int64 v14; // r12
  unsigned __int64 *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  __int64 v19[2]; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+10h] [rbp-30h]
  char v21; // [rsp+11h] [rbp-2Fh]

  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(v6 + 16);
  if ( *(_DWORD *)(v7 + 348) != 4 || (v9 = *(_DWORD *)(v7 + 352), v9 == 6) || !v9 )
  {
    v21 = 1;
    v19[0] = (__int64)".seh_* directives are not supported on this target";
    v20 = 3;
    return (__int64)sub_38BE3D0(v6, a3, (__int64)v19);
  }
  v10 = *(_QWORD *)(a1 + 72);
  if ( v10 && !*(_QWORD *)(v10 + 8) )
  {
    v21 = 1;
    v19[0] = (__int64)"Starting a function before ending the previous one!";
    v20 = 3;
    sub_38BE3D0(v6, a3, (__int64)v19);
  }
  v11 = 1;
  v12 = *(__int64 (**)())(*(_QWORD *)a1 + 16LL);
  if ( v12 != sub_38DBC10 )
    v11 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v12)(a1, a3);
  v13 = sub_22077B0(0x60u);
  v14 = v13;
  if ( v13 )
  {
    *(_QWORD *)v13 = v11;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_QWORD *)(v13 + 24) = a2;
    *(_QWORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 40) = 0;
    *(_QWORD *)(v13 + 48) = 0;
    *(_WORD *)(v13 + 56) = 0;
    *(_DWORD *)(v13 + 60) = -1;
    *(_QWORD *)(v13 + 64) = 0;
    *(_QWORD *)(v13 + 72) = 0;
    *(_QWORD *)(v13 + 80) = 0;
    *(_QWORD *)(v13 + 88) = 0;
  }
  v19[0] = v13;
  v15 = *(unsigned __int64 **)(a1 + 56);
  if ( v15 == *(unsigned __int64 **)(a1 + 64) )
  {
    sub_38E0390((unsigned __int64 *)(a1 + 48), v15, v19);
    v14 = v19[0];
LABEL_19:
    if ( v14 )
    {
      v18 = *(_QWORD *)(v14 + 72);
      if ( v18 )
        j_j___libc_free_0(v18);
      j_j___libc_free_0(v14);
    }
    goto LABEL_15;
  }
  if ( !v15 )
  {
    *(_QWORD *)(a1 + 56) = 8;
    goto LABEL_19;
  }
  *v15 = v13;
  *(_QWORD *)(a1 + 56) += 8LL;
LABEL_15:
  v16 = 0;
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 56) - 8LL);
  result = *(unsigned int *)(a1 + 120);
  *(_QWORD *)(a1 + 72) = v17;
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 112) + 32 * result;
    v16 = *(_QWORD *)(result - 32);
  }
  *(_QWORD *)(v17 + 48) = v16;
  return result;
}
