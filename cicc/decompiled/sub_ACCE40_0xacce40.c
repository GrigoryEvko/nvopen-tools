// Function: sub_ACCE40
// Address: 0xacce40
//
__int64 __fastcall sub_ACCE40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 result; // rax
  __int64 v9; // r15
  __int64 *v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 *v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]

  v5 = a2;
  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v19 = 0;
  result = 3 * v6;
  v18 = -1;
  v9 = v7 + 8 * result;
  if ( v7 == v9 )
    goto LABEL_11;
  v10 = &v18;
  do
  {
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_4;
      v11 = v19;
      *(_DWORD *)(v7 + 8) = v19;
      if ( v11 > 0x40 )
        break;
      result = v18;
      *(_QWORD *)v7 = v18;
LABEL_4:
      v7 += 24;
      if ( v9 == v7 )
        goto LABEL_8;
    }
    v12 = v7;
    v7 += 24;
    v17 = v10;
    result = sub_C43780(v12, v10);
    v10 = v17;
  }
  while ( v9 != v7 );
LABEL_8:
  if ( v19 > 0x40 && v18 )
    result = j_j___libc_free_0_0(v18);
LABEL_11:
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        result = *(unsigned int *)(v5 + 8);
        if ( (_DWORD)result || *(_QWORD *)v5 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v5 += 24;
        if ( a3 == v5 )
          return result;
      }
      sub_AC64E0(a1, v5, &v18);
      v13 = v18;
      if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
        j_j___libc_free_0_0(*(_QWORD *)v18);
      *(_QWORD *)v13 = *(_QWORD *)v5;
      *(_DWORD *)(v13 + 8) = *(_DWORD *)(v5 + 8);
      v14 = *(_QWORD *)(v5 + 16);
      result = v18;
      *(_DWORD *)(v5 + 8) = 0;
      *(_QWORD *)(result + 16) = v14;
      *(_QWORD *)(v5 + 16) = 0;
      ++*(_DWORD *)(a1 + 16);
      v15 = *(_QWORD *)(v5 + 16);
      if ( v15 )
      {
        if ( *(_DWORD *)(v15 + 32) > 0x40u )
        {
          v16 = *(_QWORD *)(v15 + 24);
          if ( v16 )
            j_j___libc_free_0_0(v16);
        }
        sub_BD7260(v15);
        result = sub_BD2DD0(v15);
      }
      if ( *(_DWORD *)(v5 + 8) > 0x40u )
      {
        if ( *(_QWORD *)v5 )
          result = j_j___libc_free_0_0(*(_QWORD *)v5);
      }
      v5 += 24;
    }
    while ( a3 != v5 );
  }
  return result;
}
