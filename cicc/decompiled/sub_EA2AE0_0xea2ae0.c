// Function: sub_EA2AE0
// Address: 0xea2ae0
//
_QWORD *__fastcall sub_EA2AE0(_QWORD *a1)
{
  __int64 v1; // rbx
  unsigned __int64 *v3; // rcx
  __int64 *v4; // rdi
  unsigned __int64 v5; // rsi
  _QWORD *result; // rax
  __int64 i; // [rsp+8h] [rbp-78h]
  _QWORD v8[2]; // [rsp+10h] [rbp-70h] BYREF
  const char *v9; // [rsp+20h] [rbp-60h] BYREF
  char v10; // [rsp+40h] [rbp-40h]
  char v11; // [rsp+41h] [rbp-3Fh]

  v1 = a1[47];
  for ( i = a1[46]; i != v1; result = sub_C91CB0(v4, v5, 3, (__int64)&v9, (__int64)v8, 1, 0, 0, 1u) )
  {
    v3 = *(unsigned __int64 **)(v1 - 8);
    v4 = (__int64 *)a1[31];
    v11 = 1;
    v1 -= 8;
    v9 = "while in macro instantiation";
    v10 = 3;
    v5 = *v3;
    v8[0] = 0;
    v8[1] = 0;
  }
  return result;
}
