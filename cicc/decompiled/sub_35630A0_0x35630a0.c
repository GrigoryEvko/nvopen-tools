// Function: sub_35630A0
// Address: 0x35630a0
//
void __fastcall sub_35630A0(__int64 a1)
{
  __int64 v1; // rax
  char *v2; // r12
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h]
  unsigned __int64 v11; // [rsp+10h] [rbp-30h]

  v1 = *(unsigned int *)(a1 + 8);
  v2 = *(char **)a1;
  v3 = *(_QWORD *)a1 + 88 * v1;
  sub_354AB10(&v9, *(_QWORD *)a1, 0x2E8BA2E8BA2E8BA3LL * ((88 * v1) >> 3));
  if ( v11 )
    sub_3562FA0(v2, v3, v11, v10);
  else
    sub_35598E0(v2, v3);
  v4 = v11;
  v5 = v11 + 88 * v10;
  if ( v11 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v4 + 32);
      if ( v6 != v4 + 48 )
        _libc_free(v6);
      v7 = *(unsigned int *)(v4 + 24);
      v8 = *(_QWORD *)(v4 + 8);
      v4 += 88;
      sub_C7D6A0(v8, 8 * v7, 8);
    }
    while ( v5 != v4 );
    v5 = v11;
  }
  j_j___libc_free_0(v5);
}
