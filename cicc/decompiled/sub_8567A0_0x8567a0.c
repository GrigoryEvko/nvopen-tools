// Function: sub_8567A0
// Address: 0x8567a0
//
void __fastcall sub_8567A0(__int64 *a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  _QWORD *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 *v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // rdi
  __int64 i; // [rsp+8h] [rbp-58h]
  _QWORD *v12; // [rsp+18h] [rbp-48h]
  unsigned __int64 v13; // [rsp+20h] [rbp-40h]
  _QWORD *v14; // [rsp+28h] [rbp-38h]

  v1 = (_QWORD *)a1[7];
  v14 = (_QWORD *)a1[6];
  v2 = (_QWORD *)(a1[5] + 8);
  v12 = (_QWORD *)a1[4];
  v3 = (_QWORD *)a1[2];
  v13 = a1[9];
  for ( i = a1[5]; v13 > (unsigned __int64)v2; ++v2 )
  {
    v4 = (_QWORD *)*v2;
    v5 = *v2 + 512LL;
    do
    {
      if ( (_QWORD *)*v4 != v4 + 2 )
        j_j___libc_free_0(*v4, v4[2] + 1LL);
      v4 += 4;
    }
    while ( (_QWORD *)v5 != v4 );
  }
  if ( v13 == i )
  {
    while ( v14 != v3 )
    {
      if ( (_QWORD *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3, v3[2] + 1LL);
      v3 += 4;
    }
    v6 = *a1;
    if ( *a1 )
      goto LABEL_17;
  }
  else
  {
    for ( ; v12 != v3; v3 += 4 )
    {
      if ( (_QWORD *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3, v3[2] + 1LL);
    }
    for ( ; v14 != v1; v1 += 4 )
    {
      if ( (_QWORD *)*v1 != v1 + 2 )
        j_j___libc_free_0(*v1, v1[2] + 1LL);
    }
    v6 = *a1;
    if ( *a1 )
    {
LABEL_17:
      v7 = (__int64 *)a1[5];
      v8 = a1[9] + 8;
      if ( v8 > (unsigned __int64)v7 )
      {
        do
        {
          v9 = *v7++;
          j_j___libc_free_0(v9, 512);
        }
        while ( v8 > (unsigned __int64)v7 );
        v6 = *a1;
      }
      j_j___libc_free_0(v6, 8 * a1[1]);
    }
  }
}
