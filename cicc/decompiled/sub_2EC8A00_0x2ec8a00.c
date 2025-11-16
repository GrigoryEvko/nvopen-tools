// Function: sub_2EC8A00
// Address: 0x2ec8a00
//
unsigned __int64 __fastcall sub_2EC8A00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // r12
  __int64 v8; // rdx
  __int64 i; // rdx
  _QWORD *v10; // r15
  _QWORD *v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 j; // r9
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  result = sub_2FF7B70(a3);
  if ( (_BYTE)result )
  {
    v7 = *(unsigned int *)(a3 + 48);
    result = *(unsigned int *)(a1 + 24);
    if ( v7 != result )
    {
      if ( v7 >= result )
      {
        if ( v7 > *(unsigned int *)(a1 + 28) )
        {
          sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), *(unsigned int *)(a3 + 48), 4u, v5, v6);
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
    }
    v10 = *(_QWORD **)(a2 + 48);
    v11 = *(_QWORD **)(a2 + 56);
    if ( v10 != v11 )
    {
      v12 = v10[2];
      v19 = a2 + 600;
      if ( !v12 )
        goto LABEL_18;
      while ( 1 )
      {
        *(_DWORD *)(a1 + 8) += *(_DWORD *)(a3 + 288) * sub_2FF7F40(a3, *v10, v12);
        v13 = *(unsigned __int16 *)(v12 + 2);
        v14 = *(_QWORD *)(*(_QWORD *)(a3 + 192) + 176LL);
        result = v14 + 6 * v13;
        for ( j = v14 + 6 * (v13 + *(unsigned __int16 *)(v12 + 4));
              j != result;
              *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v16) += *(_DWORD *)(*(_QWORD *)(a3 + 208) + 4 * v16)
                                                           * (v17 - *(unsigned __int16 *)(result - 2)) )
        {
          v16 = *(unsigned __int16 *)result;
          v17 = *(unsigned __int16 *)(result + 2);
          result += 6LL;
        }
        v10 += 32;
        if ( v11 == v10 )
          break;
        v12 = v10[2];
        if ( !v12 )
        {
LABEL_18:
          if ( (unsigned __int8)sub_2FF7B70(v19) )
          {
            v18 = sub_2FF7DB0(v19, *v10);
            v10[2] = v18;
            v12 = v18;
          }
          else
          {
            v12 = v10[2];
          }
        }
      }
    }
  }
  return result;
}
