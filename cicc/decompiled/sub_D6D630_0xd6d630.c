// Function: sub_D6D630
// Address: 0xd6d630
//
unsigned __int64 __fastcall sub_D6D630(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r8
  _QWORD *v17; // rdi
  _QWORD v18[2]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v19; // [rsp+10h] [rbp-40h]

  v3 = a2;
  v4 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v5 = *(unsigned __int64 **)(a2 - 8);
    v6 = &v5[v4];
  }
  else
  {
    v5 = (unsigned __int64 *)(a2 - v4 * 8);
    v6 = (unsigned __int64 *)a2;
  }
  v18[0] = 0;
  v18[1] = 0;
  v19 = a2;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)v18);
  if ( *(_QWORD *)(a1 + 752) )
  {
    v15 = *(_QWORD **)(a1 + 728);
    v16 = a1 + 720;
    if ( v15 )
    {
      v17 = (_QWORD *)(a1 + 720);
      do
      {
        if ( v15[6] < v19 )
        {
          v15 = (_QWORD *)v15[3];
        }
        else
        {
          v17 = v15;
          v15 = (_QWORD *)v15[2];
        }
      }
      while ( v15 );
      if ( (_QWORD *)v16 != v17 && v19 >= v17[6] )
      {
LABEL_12:
        sub_D68D70(v18);
        return v3;
      }
    }
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 504);
    v8 = v7 + 24LL * *(unsigned int *)(a1 + 512);
    if ( v7 != v8 )
    {
      while ( *(_QWORD *)(v7 + 16) != v19 )
      {
        v7 += 24;
        if ( v8 == v7 )
          goto LABEL_16;
      }
      if ( v8 != v7 )
        goto LABEL_12;
    }
  }
LABEL_16:
  sub_D68D70(v18);
  if ( v6 == v5 )
    return *(_QWORD *)(*(_QWORD *)a1 + 128LL);
  v10 = v5;
  v11 = 0;
  do
  {
    if ( *v10 != v11 && a2 != *v10 )
    {
      if ( v11 )
        return v3;
      v11 = *v10;
    }
    v10 += 4;
  }
  while ( v6 != v10 );
  if ( !v11 )
    return *(_QWORD *)(*(_QWORD *)a1 + 128LL);
  sub_BD84D0(a2, v11);
  sub_D6E4B0(a1, a2, 0);
  return sub_D6D330(a1, v11, v12, v13, v14);
}
