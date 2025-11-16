// Function: sub_22801B0
// Address: 0x22801b0
//
__int64 __fastcall sub_22801B0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r8
  __int64 v5; // r9
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  __int64 result; // rax
  __int64 v9; // r12
  unsigned int v10; // eax
  _QWORD *v11; // rcx
  int v12; // eax
  unsigned int v13; // esi
  unsigned int v14; // edi
  _QWORD *v15; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v16; // [rsp+8h] [rbp-38h] BYREF
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  __int64 v18; // [rsp+18h] [rbp-28h]

  v17 = *a2;
  v18 = *(unsigned int *)(a1 + 88);
  if ( !(unsigned __int8)sub_227C450(a1, &v17, &v15) )
  {
    v10 = *(_DWORD *)(a1 + 8);
    v11 = v15;
    ++*(_QWORD *)a1;
    v16 = v11;
    v12 = (v10 >> 1) + 1;
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v14 = 12;
      v13 = 4;
    }
    else
    {
      v13 = *(_DWORD *)(a1 + 24);
      v14 = 3 * v13;
    }
    if ( 4 * v12 >= v14 )
    {
      v13 *= 2;
    }
    else if ( v13 - (v12 + *(_DWORD *)(a1 + 12)) > v13 >> 3 )
    {
LABEL_11:
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v12);
      if ( *v11 != -4096 )
        --*(_DWORD *)(a1 + 12);
      *v11 = v17;
      v11[1] = v18;
      goto LABEL_4;
    }
    sub_227FE50(a1, v13);
    sub_227C450(a1, &v17, &v16);
    v11 = v16;
    v12 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
    goto LABEL_11;
  }
  v6 = v15;
  v7 = v15[1];
  result = *(unsigned int *)(a1 + 88) - 1LL;
  if ( v7 == result )
    return result;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v7) = 0;
  v6[1] = *(unsigned int *)(a1 + 88);
LABEL_4:
  result = *(unsigned int *)(a1 + 88);
  v9 = *a2;
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), result + 1, 8u, v4, v5);
    result = *(unsigned int *)(a1 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * result) = v9;
  ++*(_DWORD *)(a1 + 88);
  return result;
}
