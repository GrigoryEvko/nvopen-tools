// Function: sub_1843480
// Address: 0x1843480
//
void __fastcall sub_1843480(_QWORD *a1, __int64 *a2)
{
  _QWORD *v3; // r12
  __int64 v4; // rdi
  unsigned __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned int v8; // ecx
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rax

  v3 = a1 + 1;
  v4 = a1[2];
  if ( !v4 )
  {
    v6 = (__int64)v3;
    goto LABEL_25;
  }
  v5 = *a2;
  v6 = (__int64)v3;
  v7 = v4;
  do
  {
    if ( v5 > *(_QWORD *)(v7 + 32)
      || v5 == *(_QWORD *)(v7 + 32)
      && ((v8 = *((_DWORD *)a2 + 2), *(_DWORD *)(v7 + 40) < v8)
       || *(_DWORD *)(v7 + 40) == v8 && *(_BYTE *)(v7 + 44) < *((_BYTE *)a2 + 12)) )
    {
      v7 = *(_QWORD *)(v7 + 24);
    }
    else
    {
      v6 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
  }
  while ( v7 );
  if ( v3 == (_QWORD *)v6 )
  {
LABEL_25:
    if ( v6 == a1[3] )
    {
LABEL_23:
      sub_1842750(v4);
      a1[2] = 0;
      a1[3] = v3;
      a1[4] = v3;
      a1[5] = 0;
    }
    return;
  }
  v9 = v6;
  while ( *(_QWORD *)(v9 + 32) == v5 && ((a2[1] ^ *(_QWORD *)(v9 + 40)) & 0xFFFFFFFFFFLL) == 0 )
  {
    sub_1843600(a1, v9 + 48, v5, 0xFFFFFFFFFFLL);
    v9 = sub_220EEE0(v9);
    if ( (_QWORD *)v9 == v3 )
    {
      if ( v6 == a1[3] )
      {
LABEL_22:
        v4 = a1[2];
        goto LABEL_23;
      }
      goto LABEL_15;
    }
    v5 = *a2;
  }
  if ( v6 == a1[3] && v3 == (_QWORD *)v9 )
    goto LABEL_22;
  if ( v6 == v9 )
    return;
  do
  {
LABEL_15:
    v10 = v6;
    v6 = sub_220EF30(v6);
    v11 = sub_220F330(v10, v3);
    j_j___libc_free_0(v11, 64);
    --a1[5];
  }
  while ( v9 != v6 );
}
