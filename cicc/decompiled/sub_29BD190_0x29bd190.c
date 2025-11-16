// Function: sub_29BD190
// Address: 0x29bd190
//
__int64 __fastcall sub_29BD190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 *v10; // r13
  __int64 *v11; // r12
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 *v15; // [rsp+0h] [rbp-40h]
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(__int64 **)a1;
  v16[0] = a2;
  v6 *= 8;
  v15 = (__int64 *)((char *)v7 + v6);
  v8 = v6 >> 3;
  v9 = v6 >> 5;
  if ( v9 )
  {
    v10 = &v7[4 * v9];
    while ( !sub_29BCF80(v16, v7) )
    {
      v11 = v7++;
      if ( sub_29BCF80(v16, v7) )
        break;
      v7 = v11 + 2;
      if ( sub_29BCF80(v16, v11 + 2) )
        break;
      v7 = v11 + 3;
      if ( sub_29BCF80(v16, v11 + 3) )
        break;
      v7 = v11 + 4;
      if ( v10 == v11 + 4 )
      {
        v8 = v15 - v7;
        goto LABEL_11;
      }
    }
LABEL_8:
    result = 0;
    if ( v15 != v7 )
      return result;
    goto LABEL_14;
  }
LABEL_11:
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      if ( v8 != 1 )
        goto LABEL_14;
      goto LABEL_21;
    }
    if ( sub_29BCF80(v16, v7) )
      goto LABEL_8;
    ++v7;
  }
  if ( sub_29BCF80(v16, v7) )
    goto LABEL_8;
  ++v7;
LABEL_21:
  if ( sub_29BCF80(v16, v7) )
    goto LABEL_8;
LABEL_14:
  v13 = *(unsigned int *)(a1 + 8);
  v14 = v16[0];
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 8u, a5, a6);
    v13 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v13) = v14;
  ++*(_DWORD *)(a1 + 8);
  return 1;
}
