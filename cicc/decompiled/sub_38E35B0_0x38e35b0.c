// Function: sub_38E35B0
// Address: 0x38e35b0
//
_QWORD *__fastcall sub_38E35B0(_QWORD *a1)
{
  unsigned __int64 **v1; // rbx
  __int64 *v3; // rdi
  unsigned __int64 v4; // rsi
  _QWORD *result; // rax
  unsigned __int64 **i; // [rsp+8h] [rbp-68h]
  unsigned __int64 v7[2]; // [rsp+10h] [rbp-60h] BYREF
  const char *v8; // [rsp+20h] [rbp-50h] BYREF
  char v9; // [rsp+30h] [rbp-40h]
  char v10; // [rsp+31h] [rbp-3Fh]

  v1 = (unsigned __int64 **)a1[57];
  for ( i = (unsigned __int64 **)a1[56]; i != v1; result = sub_16D14E0(v3, v4, 3, (__int64)&v8, v7, 1, 0, 0, 1u) )
  {
    v10 = 1;
    v8 = "while in macro instantiation";
    --v1;
    v9 = 3;
    v3 = (__int64 *)a1[43];
    v4 = **v1;
    v7[0] = 0;
    v7[1] = 0;
  }
  return result;
}
