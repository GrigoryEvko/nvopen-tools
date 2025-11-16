// Function: sub_2919EF0
// Address: 0x2919ef0
//
_BYTE *__fastcall sub_2919EF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *result; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  _QWORD *v9; // rcx
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 *v12; // rdx

  result = (_BYTE *)((unsigned int)*(unsigned __int8 *)(a2 + 8) - 17);
  if ( (unsigned int)result <= 1 )
  {
    v7 = *a1;
    v8 = *(unsigned int *)(*a1 + 8);
    if ( (_DWORD)v8 )
    {
      v10 = **(_QWORD **)v7;
      v11 = sub_9208B0(a1[1], a2);
      if ( v11 != sub_9208B0(a1[1], v10) )
      {
        result = (_BYTE *)*a1;
        *(_DWORD *)(*a1 + 8) = 0;
        return result;
      }
      v7 = *a1;
      v8 = *(unsigned int *)(*a1 + 8);
    }
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 12) )
    {
      sub_C8D5F0(v7, (const void *)(v7 + 16), v8 + 1, 8u, a5, a6);
      v8 = *(unsigned int *)(v7 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v7 + 8 * v8) = a2;
    ++*(_DWORD *)(v7 + 8);
    v9 = (_QWORD *)a1[2];
    result = *(_BYTE **)(a2 + 24);
    if ( *v9 )
    {
      if ( result != (_BYTE *)*v9 )
        *(_BYTE *)a1[3] = 0;
    }
    else
    {
      *v9 = result;
    }
    if ( result[8] == 14 )
    {
      *(_BYTE *)a1[4] = 1;
      v12 = (__int64 *)a1[5];
      result = (_BYTE *)*v12;
      if ( *v12 )
      {
        if ( (_BYTE *)a2 != result )
        {
          result = (_BYTE *)a1[6];
          *result = 0;
        }
      }
      else
      {
        *v12 = a2;
      }
    }
  }
  return result;
}
