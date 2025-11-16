// Function: sub_28EDAB0
// Address: 0x28edab0
//
void __fastcall sub_28EDAB0(unsigned __int64 *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // rbx
  _QWORD **v3; // r12
  _QWORD *v4; // r14
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r12
  unsigned __int64 *v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  unsigned __int64 i; // [rsp+8h] [rbp-58h]
  _QWORD *v16; // [rsp+18h] [rbp-48h]
  unsigned __int64 v17; // [rsp+20h] [rbp-40h]
  _QWORD *v18; // [rsp+28h] [rbp-38h]

  v1 = (_QWORD *)a1[7];
  v2 = (_QWORD *)a1[2];
  v16 = (_QWORD *)a1[4];
  v18 = (_QWORD *)a1[6];
  v17 = a1[9];
  v3 = (_QWORD **)(a1[5] + 8);
  for ( i = a1[5]; v17 > (unsigned __int64)v3; ++v3 )
  {
    v4 = *v3;
    v5 = (__int64)(*v3 + 63);
    do
    {
      v6 = v4[2];
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD60C0(v4);
      v4 += 3;
    }
    while ( (_QWORD *)v5 != v4 );
  }
  if ( i != v17 )
  {
    for ( ; v16 != v2; v2 += 3 )
    {
      v7 = v2[2];
      if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
        sub_BD60C0(v2);
    }
    for ( ; v18 != v1; v1 += 3 )
    {
      v8 = v1[2];
      if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
        sub_BD60C0(v1);
    }
LABEL_19:
    v9 = *a1;
    if ( !*a1 )
      return;
    goto LABEL_20;
  }
  if ( v2 == v18 )
    goto LABEL_19;
  do
  {
    v13 = v2[2];
    if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
      sub_BD60C0(v2);
    v2 += 3;
  }
  while ( v18 != v2 );
  v9 = *a1;
  if ( *a1 )
  {
LABEL_20:
    v10 = a1[9] + 8;
    v11 = (unsigned __int64 *)a1[5];
    if ( v10 > (unsigned __int64)v11 )
    {
      do
      {
        v12 = *v11++;
        j_j___libc_free_0(v12);
      }
      while ( v10 > (unsigned __int64)v11 );
      v9 = *a1;
    }
    j_j___libc_free_0(v9);
  }
}
