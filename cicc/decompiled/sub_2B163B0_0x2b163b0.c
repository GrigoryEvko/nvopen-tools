// Function: sub_2B163B0
// Address: 0x2b163b0
//
__int64 __fastcall sub_2B163B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // r13
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax
  unsigned __int8 *v10; // r15
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 *v14; // r12
  __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // esi
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2);
  if ( *(_BYTE *)v2 == 13 )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  if ( *v3 )
  {
LABEL_3:
    v4 = *(_QWORD *)(v2 + 16);
    if ( !v4 )
      goto LABEL_4;
    goto LABEL_21;
  }
  if ( **(_DWORD **)(a1 + 16) == 61 )
  {
    *v3 = *(_QWORD *)(*(_QWORD *)(v2 - 64) + 8LL);
    goto LABEL_3;
  }
  v16 = *(_QWORD *)(*(_QWORD *)(v2 - 32) + 8LL);
  if ( *(_BYTE *)(v16 + 8) == 16 )
    v17 = *(_DWORD *)(v16 + 32);
  else
    v17 = *(_DWORD *)(v16 + 12);
  *v3 = sub_2B08680(**(_QWORD **)(a1 + 24), v17);
  v4 = *(_QWORD *)(v2 + 16);
  if ( !v4 )
    goto LABEL_4;
LABEL_21:
  if ( *(_QWORD *)(v4 + 8) )
    goto LABEL_4;
  v10 = *(unsigned __int8 **)(v4 + 24);
  if ( (unsigned __int8)(*v10 - 68) > 1u )
    goto LABEL_4;
  v11 = *((_QWORD *)v10 + 2);
  if ( v11 )
  {
    while ( **(_BYTE **)(v11 + 24) == 63 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        goto LABEL_26;
    }
LABEL_4:
    v5 = *(_QWORD *)(a1 + 48);
    v6 = *(_DWORD *)(v5 + 8);
    if ( v6 <= 0x40 )
    {
      if ( !*(_QWORD *)v5 )
      {
        v7 = **(_QWORD **)(a1 + 8);
        if ( *(_BYTE *)(v7 + 8) == 17 )
        {
LABEL_7:
          v19 = *(_DWORD *)(v7 + 32);
          if ( v19 > 0x40 )
          {
            sub_C43690((__int64)&v18, 0, 0);
            v5 = *(_QWORD *)(a1 + 48);
            goto LABEL_9;
          }
LABEL_20:
          v18 = 0;
LABEL_9:
          if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
            j_j___libc_free_0_0(*(_QWORD *)v5);
          *(_QWORD *)v5 = v18;
          *(_DWORD *)(v5 + 8) = v19;
          v5 = *(_QWORD *)(a1 + 48);
          goto LABEL_13;
        }
LABEL_19:
        v19 = 1;
        goto LABEL_20;
      }
    }
    else if ( v6 == (unsigned int)sub_C444A0(*(_QWORD *)(a1 + 48)) )
    {
      v7 = **(_QWORD **)(a1 + 8);
      if ( *(_BYTE *)(v7 + 8) == 17 )
        goto LABEL_7;
      goto LABEL_19;
    }
LABEL_13:
    v18 = sub_2B15730(v2);
    v8 = 1LL << v18;
    if ( *(_DWORD *)(v5 + 8) > 0x40u )
      *(_QWORD *)(*(_QWORD *)v5 + 8LL * ((unsigned int)v18 >> 6)) |= v8;
    else
      *(_QWORD *)v5 |= v8;
    return 0;
  }
LABEL_26:
  v18 = sub_2B15730(v2);
  v13 = sub_DFD220(v12);
  v14 = *(__int64 **)(*(_QWORD *)(a1 + 32) + 3296LL);
  sub_DFBCC0(v10);
  v15 = sub_DFD060(v14, (unsigned int)*v10 - 29, *((_QWORD *)v10 + 1), *(_QWORD *)(v2 + 8));
  result = v13 - v15;
  if ( __OFSUB__(v13, v15) )
  {
    result = 0x8000000000000000LL;
    if ( v15 <= 0 )
      return 0x7FFFFFFFFFFFFFFFLL;
  }
  return result;
}
