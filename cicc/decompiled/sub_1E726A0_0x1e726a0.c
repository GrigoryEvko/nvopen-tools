// Function: sub_1E726A0
// Address: 0x1e726a0
//
unsigned __int64 __fastcall sub_1E726A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  int v5; // r8d
  int v6; // r9d
  unsigned __int64 v7; // r15
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 j; // r8
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  result = sub_1F4B670(a3);
  if ( !(_BYTE)result )
    return result;
  v7 = *(unsigned int *)(a3 + 48);
  result = *(unsigned int *)(a1 + 24);
  if ( v7 >= result )
  {
    if ( v7 <= result )
      goto LABEL_12;
    if ( v7 > *(unsigned int *)(a1 + 28) )
    {
      sub_16CD150(a1 + 16, (const void *)(a1 + 32), *(unsigned int *)(a3 + 48), 4, v5, v6);
      result = *(unsigned int *)(a1 + 24);
    }
    v8 = *(_QWORD *)(a1 + 16);
    result = v8 + 4 * result;
    for ( i = v8 + 4 * v7; i != result; result += 4LL )
    {
      if ( result )
        *(_DWORD *)result = 0;
    }
  }
  *(_DWORD *)(a1 + 24) = v7;
LABEL_12:
  v10 = *(_QWORD *)(a2 + 48);
  v11 = *(_QWORD *)(a2 + 56);
  if ( v10 != v11 )
  {
    v12 = *(_QWORD *)(v10 + 24);
    v18 = a2 + 632;
    if ( !v12 )
      goto LABEL_18;
    while ( 1 )
    {
      *(_DWORD *)(a1 + 8) += *(_DWORD *)(a3 + 272) * sub_1F4BA40(a3, *(_QWORD *)(v10 + 8), v12);
      v13 = *(unsigned __int16 *)(v12 + 2);
      v14 = *(_QWORD *)(*(_QWORD *)(a3 + 176) + 136LL);
      result = v14 + 4 * v13;
      for ( j = v14 + 4 * (v13 + *(unsigned __int16 *)(v12 + 4));
            j != result;
            *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v16) += *(_DWORD *)(*(_QWORD *)(a3 + 192) + 4 * v16)
                                                         * *(unsigned __int16 *)(result - 2) )
      {
        v16 = *(unsigned __int16 *)result;
        result += 4LL;
      }
      v10 += 272;
      if ( v11 == v10 )
        break;
      v12 = *(_QWORD *)(v10 + 24);
      if ( !v12 )
      {
LABEL_18:
        if ( (unsigned __int8)sub_1F4B670(v18) )
        {
          v17 = sub_1F4B8B0(v18, *(_QWORD *)(v10 + 8));
          *(_QWORD *)(v10 + 24) = v17;
          v12 = v17;
        }
        else
        {
          v12 = *(_QWORD *)(v10 + 24);
        }
      }
    }
  }
  return result;
}
