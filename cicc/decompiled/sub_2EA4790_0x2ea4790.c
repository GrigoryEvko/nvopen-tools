// Function: sub_2EA4790
// Address: 0x2ea4790
//
__int64 __fastcall sub_2EA4790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r8
  __int64 *v8; // r13
  __int64 *v9; // r12
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  result = *(_QWORD *)(a1 + 40);
  v14 = result;
  v15 = *(_QWORD *)(a1 + 32);
  if ( result != v15 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)v15 + 112LL);
      v8 = (__int64 *)(v7 + 8LL * *(unsigned int *)(*(_QWORD *)v15 + 120LL));
      v9 = (__int64 *)v7;
      if ( (__int64 *)v7 != v8 )
        break;
LABEL_9:
      v15 += 8;
      result = v15;
      if ( v14 == v15 )
        return result;
    }
    while ( 1 )
    {
      v10 = *v9;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v11 = *(_QWORD **)(a1 + 64);
        v12 = &v11[*(unsigned int *)(a1 + 76)];
        if ( v11 == v12 )
          goto LABEL_12;
        while ( v10 != *v11 )
        {
          if ( v12 == ++v11 )
            goto LABEL_12;
        }
LABEL_8:
        if ( v8 == ++v9 )
          goto LABEL_9;
      }
      else
      {
        if ( sub_C8CA60(a1 + 56, *v9) )
          goto LABEL_8;
LABEL_12:
        v13 = *(unsigned int *)(a2 + 8);
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 8u, v7, a6);
          v13 = *(unsigned int *)(a2 + 8);
        }
        ++v9;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = v10;
        ++*(_DWORD *)(a2 + 8);
        if ( v8 == v9 )
          goto LABEL_9;
      }
    }
  }
  return result;
}
