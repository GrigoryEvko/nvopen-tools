// Function: sub_1B2EEE0
// Address: 0x1b2eee0
//
__int64 __fastcall sub_1B2EEE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v8; // rax
  __int64 v9; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rcx
  __int64 v13; // r12
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 *v16; // rsi
  __int64 *v17; // rcx

  v8 = *(__int64 **)(a2 + 8);
  if ( *(__int64 **)(a2 + 16) != v8 )
    goto LABEL_2;
  v15 = *(unsigned int *)(a2 + 28);
  v16 = &v8[v15];
  if ( v8 == v16 )
  {
LABEL_14:
    if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 24) )
    {
LABEL_2:
      sub_16CCBA0(a2, a3);
      goto LABEL_3;
    }
    *(_DWORD *)(a2 + 28) = v15 + 1;
    *v16 = a3;
    ++*(_QWORD *)a2;
  }
  else
  {
    v17 = 0;
    while ( a3 != *v8 )
    {
      if ( *v8 == -2 )
        v17 = v8;
      if ( v16 == ++v8 )
      {
        if ( !v17 )
          goto LABEL_14;
        *v17 = a3;
        --*(_DWORD *)(a2 + 32);
        ++*(_QWORD *)a2;
        break;
      }
    }
  }
LABEL_3:
  v9 = sub_1B2EB40((__int64)a1, a3);
  v12 = *a1;
  *(_QWORD *)(a4 + 16) = a1;
  v13 = v9;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a4 + 8) = v12 | *(_QWORD *)(a4 + 8) & 7LL;
  *(_QWORD *)(v12 + 8) = a4 + 8;
  *a1 = *a1 & 7 | (a4 + 8);
  result = *(unsigned int *)(v9 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(v13 + 12) )
  {
    sub_16CD150(v13, (const void *)(v13 + 16), 0, 8, v10, v11);
    result = *(unsigned int *)(v13 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v13 + 8 * result) = a4;
  ++*(_DWORD *)(v13 + 8);
  return result;
}
