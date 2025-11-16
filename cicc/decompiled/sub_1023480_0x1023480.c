// Function: sub_1023480
// Address: 0x1023480
//
__int64 __fastcall sub_1023480(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 *v12; // r12
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 *v16; // r12
  __int64 v17; // rdi

  *(_QWORD *)a1 = 6;
  *(_QWORD *)(a1 + 8) = 0;
  if ( a2 )
  {
    *(_QWORD *)(a1 + 16) = a2;
    if ( a2 != -4096 && a2 != -8192 )
      sub_BD73F0(a1);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
  }
  result = 0x200000000LL;
  *(_DWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( a6 )
  {
    v12 = *(__int64 **)a6;
    result = *(unsigned int *)(a6 + 8);
    v13 = *(_QWORD *)a6 + 8 * result;
    if ( v13 != *(_QWORD *)a6 )
    {
      v14 = *v12;
      v15 = a1 + 64;
      v16 = v12 + 1;
      result = 0;
      v17 = a1 + 48;
      while ( 1 )
      {
        *(_QWORD *)(v15 + 8 * result) = v14;
        result = (unsigned int)(*(_DWORD *)(a1 + 56) + 1);
        *(_DWORD *)(a1 + 56) = result;
        if ( (__int64 *)v13 == v16 )
          break;
        v14 = *v16;
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
        {
          sub_C8D5F0(v17, (const void *)(a1 + 64), result + 1, 8u, a5, a6);
          result = *(unsigned int *)(a1 + 56);
        }
        v15 = *(_QWORD *)(a1 + 48);
        ++v16;
      }
    }
  }
  return result;
}
