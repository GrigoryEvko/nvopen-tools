// Function: sub_14439E0
// Address: 0x14439e0
//
__int64 __fastcall sub_14439E0(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 *v6; // rbx
  __int64 *i; // r12
  __int64 v8; // rdi

  v2 = a1[10];
  while ( v2 )
  {
    v3 = v2;
    sub_1442C80(*(_QWORD **)(v2 + 24));
    v4 = *(_QWORD *)(v2 + 40);
    v2 = *(_QWORD *)(v2 + 16);
    if ( v4 )
      j_j___libc_free_0(v4, 16);
    j_j___libc_free_0(v3, 48);
  }
  result = (__int64)(a1 + 9);
  v6 = (__int64 *)a1[5];
  a1[10] = 0;
  a1[11] = a1 + 9;
  a1[12] = a1 + 9;
  a1[13] = 0;
  for ( i = (__int64 *)a1[6]; i != v6; result = sub_14439E0(v8) )
    v8 = *v6++;
  return result;
}
