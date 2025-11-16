// Function: sub_1E63E80
// Address: 0x1e63e80
//
_QWORD *__fastcall sub_1E63E80(__int64 *a1)
{
  __int64 v1; // rbp
  _QWORD *result; // rax
  __int64 v3; // rsi
  __int64 i; // rbx
  __int64 v5; // rdi
  __int64 v6; // [rsp-48h] [rbp-48h] BYREF
  int v7; // [rsp-40h] [rbp-40h] BYREF
  __int64 v8; // [rsp-38h] [rbp-38h]
  int *v9; // [rsp-30h] [rbp-30h]
  int *v10; // [rsp-28h] [rbp-28h]
  __int64 v11; // [rsp-20h] [rbp-20h]
  __int64 v12; // [rsp-8h] [rbp-8h]

  result = byte_4FC71FC;
  if ( byte_4FC71FC[0] )
  {
    v12 = v1;
    v3 = *a1;
    v7 = 0;
    v8 = 0;
    v9 = &v7;
    v10 = &v7;
    v11 = 0;
    result = sub_1E63DB0(a1, (_QWORD *)(v3 & 0xFFFFFFFFFFFFFFF8LL), &v6);
    for ( i = v8; i; result = (_QWORD *)j_j___libc_free_0(v5, 40) )
    {
      sub_1E62060(*(_QWORD *)(i + 24));
      v5 = i;
      i = *(_QWORD *)(i + 16);
    }
  }
  return result;
}
