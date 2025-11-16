// Function: sub_159BF50
// Address: 0x159bf50
//
__int64 __fastcall sub_159BF50(__int64 a1, __int64 a2, __int64 a3)
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
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 *v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-38h]

  v5 = a2;
  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v18 = 0;
  result = 3 * v6;
  v17 = 0;
  v9 = v7 + 8 * result;
  if ( v7 == v9 )
    goto LABEL_11;
  v10 = &v17;
  do
  {
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_4;
      v11 = v18;
      *(_DWORD *)(v7 + 8) = v18;
      if ( v11 > 0x40 )
        break;
      result = v17;
      *(_QWORD *)v7 = v17;
LABEL_4:
      v7 += 24;
      if ( v9 == v7 )
        goto LABEL_8;
    }
    v12 = v7;
    v7 += 24;
    v16 = v10;
    result = sub_16A4FD0(v12, v10);
    v10 = v16;
  }
  while ( v9 != v7 );
LABEL_8:
  if ( v18 > 0x40 && v17 )
    result = j_j___libc_free_0_0(v17);
LABEL_11:
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        result = *(unsigned int *)(v5 + 8);
        if ( (_DWORD)result || *(_QWORD *)v5 > 1u )
          break;
        v5 += 24;
        if ( a3 == v5 )
          return result;
      }
      sub_1598260(a1, v5, &v17);
      v13 = v17;
      if ( *(_DWORD *)(v17 + 8) > 0x40u && *(_QWORD *)v17 )
        j_j___libc_free_0_0(*(_QWORD *)v17);
      *(_QWORD *)v13 = *(_QWORD *)v5;
      *(_DWORD *)(v13 + 8) = *(_DWORD *)(v5 + 8);
      result = *(_QWORD *)(v5 + 16);
      *(_DWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v13 + 16) = result;
      *(_QWORD *)(v5 + 16) = 0;
      ++*(_DWORD *)(a1 + 16);
      v14 = *(_QWORD *)(v5 + 16);
      if ( v14 )
      {
        if ( *(_DWORD *)(v14 + 32) > 0x40u )
        {
          v15 = *(_QWORD *)(v14 + 24);
          if ( v15 )
            j_j___libc_free_0_0(v15);
        }
        sub_164BE60(v14);
        result = sub_1648B90(v14);
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
