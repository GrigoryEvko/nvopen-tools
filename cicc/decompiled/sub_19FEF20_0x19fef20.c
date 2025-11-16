// Function: sub_19FEF20
// Address: 0x19fef20
//
__int64 __fastcall sub_19FEF20(__int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v7; // r13
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rax
  const char *v11; // [rsp-38h] [rbp-38h] BYREF
  char v12; // [rsp-28h] [rbp-28h]
  char v13; // [rsp-27h] [rbp-27h]

  v2 = *((unsigned int *)a2 + 2);
  v3 = *a2;
  if ( (_DWORD)v2 == 1 )
    return *(_QWORD *)(v3 + 16);
  v5 = (unsigned int)(v2 - 1);
  v7 = *(_QWORD *)(v3 + 24 * v2 - 8);
  *((_DWORD *)a2 + 2) = v5;
  v8 = (_QWORD *)(v3 + 24 * v5);
  v9 = v8[2];
  if ( v9 != 0 && v9 != -8 && v9 != -16 )
    sub_1649B30(v8);
  v10 = (__int64 *)sub_19FEF20(a1, a2);
  v13 = 1;
  v11 = "reass.add";
  v12 = 3;
  return sub_19FE280(v10, v7, (__int64)&v11, a1, a1);
}
