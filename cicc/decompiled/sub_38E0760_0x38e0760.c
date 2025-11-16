// Function: sub_38E0760
// Address: 0x38e0760
//
__int64 __fastcall sub_38E0760(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 (*v6)(); // rax
  __int64 v7; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // r12
  unsigned __int64 *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14[7]; // [rsp+8h] [rbp-38h] BYREF

  result = sub_38DD280(a1, a2);
  if ( !result )
    return result;
  v4 = result;
  v5 = 1;
  v6 = *(__int64 (**)())(*(_QWORD *)a1 + 16LL);
  if ( v6 != sub_38DBC10 )
    v5 = ((__int64 (__fastcall *)(__int64))v6)(a1);
  v7 = *(_QWORD *)(v4 + 24);
  v8 = sub_22077B0(0x60u);
  v9 = v8;
  if ( v8 )
  {
    *(_QWORD *)v8 = v5;
    *(_QWORD *)(v8 + 8) = 0;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = v7;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_WORD *)(v8 + 56) = 0;
    *(_DWORD *)(v8 + 60) = -1;
    *(_QWORD *)(v8 + 64) = v4;
    *(_QWORD *)(v8 + 72) = 0;
    *(_QWORD *)(v8 + 80) = 0;
    *(_QWORD *)(v8 + 88) = 0;
  }
  v14[0] = v8;
  v10 = *(unsigned __int64 **)(a1 + 56);
  if ( v10 == *(unsigned __int64 **)(a1 + 64) )
  {
    sub_38E0390((unsigned __int64 *)(a1 + 48), v10, (__int64 *)v14);
    v9 = v14[0];
LABEL_14:
    if ( v9 )
    {
      v13 = *(_QWORD *)(v9 + 72);
      if ( v13 )
        j_j___libc_free_0(v13);
      j_j___libc_free_0(v9);
    }
    goto LABEL_9;
  }
  if ( !v10 )
  {
    *(_QWORD *)(a1 + 56) = 8;
    goto LABEL_14;
  }
  *v10 = v8;
  *(_QWORD *)(a1 + 56) += 8LL;
LABEL_9:
  v11 = 0;
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 56) - 8LL);
  result = *(unsigned int *)(a1 + 120);
  *(_QWORD *)(a1 + 72) = v12;
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 112) + 32 * result;
    v11 = *(_QWORD *)(result - 32);
  }
  *(_QWORD *)(v12 + 48) = v11;
  return result;
}
