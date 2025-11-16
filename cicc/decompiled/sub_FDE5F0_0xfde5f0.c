// Function: sub_FDE5F0
// Address: 0xfde5f0
//
__int64 __fastcall sub_FDE5F0(_QWORD *a1)
{
  __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // rax
  _DWORD *v5; // rdi
  _QWORD *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 *v11; // rax
  __int64 *v12; // rdx
  __int64 result; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  unsigned int v16[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a1[8];
  v3 = *(__int64 **)(v2 + 8);
  if ( !v3 )
    goto LABEL_4;
  v4 = *((unsigned int *)v3 + 3);
  v5 = (_DWORD *)v3[12];
  if ( (unsigned int)v4 > 1 )
  {
    if ( !sub_FDC990(v5, &v5[v4], (_DWORD *)v2) )
      goto LABEL_4;
  }
  else if ( *(_DWORD *)v2 != *v5 )
  {
LABEL_4:
    v6 = (_QWORD *)(v2 + 16);
    goto LABEL_5;
  }
  if ( !*((_BYTE *)v3 + 8) )
    goto LABEL_4;
  v14 = *v3;
  if ( !*v3
    || (v15 = *(unsigned int *)(v14 + 12), (unsigned int)v15 <= 1)
    || !sub_FDC990(*(_DWORD **)(v14 + 96), (_DWORD *)(*(_QWORD *)(v14 + 96) + 4 * v15), (_DWORD *)v2)
    || (v6 = (_QWORD *)(v14 + 152), !*(_BYTE *)(v14 + 8)) )
  {
    v6 = v3 + 19;
  }
LABEL_5:
  *v6 = -1;
  v7 = a1[17];
  v8 = a1[18];
  if ( v7 == v8 )
    return 1;
  v9 = a1[17];
  while ( 1 )
  {
    v16[0] = (v9 - v7) >> 3;
    v10 = a1[8] + 24LL * v16[0];
    v11 = *(__int64 **)(v10 + 8);
    if ( !v11 || !*((_BYTE *)v11 + 8) )
      goto LABEL_27;
    do
    {
      v12 = v11;
      v11 = (__int64 *)*v11;
    }
    while ( v11 && *((_BYTE *)v11 + 8) );
    if ( *(_DWORD *)v10 == *(_DWORD *)v12[12] )
    {
LABEL_27:
      result = sub_FDE3B0(a1, 0, v16);
      if ( !(_BYTE)result )
        return result;
    }
    v9 += 8;
    if ( v8 == v9 )
      return 1;
    v7 = a1[17];
  }
}
