// Function: sub_C43690
// Address: 0xc43690
//
__int64 __fastcall sub_C43690(__int64 a1, __int64 a2, char a3)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r14
  unsigned int v9; // r13d
  int v10; // ebx
  __int64 result; // rax
  _QWORD *v12; // rcx

  v4 = ((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6;
  v5 = 8 * v4;
  if ( a2 < 0 && a3 )
  {
    v6 = (__int64 *)sub_2207820(v5);
    v7 = *(unsigned int *)(a1 + 8);
    *v6 = a2;
    v8 = v6;
    v9 = v7;
    *(_QWORD *)a1 = v6;
    v10 = ((unsigned __int64)(v7 + 63) >> 6) - 1;
    memset(v6 + 1, 255, (unsigned int)(8 * v10));
    result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
    if ( v9 )
    {
      if ( v9 > 0x40 )
      {
        v8[v10] &= result;
        return result;
      }
      result &= (unsigned __int64)v8;
    }
    else
    {
      result = 0;
    }
    *(_QWORD *)a1 = result;
  }
  else
  {
    result = sub_2207820(v5);
    v12 = (_QWORD *)result;
    if ( result )
    {
      if ( v4 )
      {
        result = (__int64)memset((void *)result, 0, 8 * v4);
        v12 = (_QWORD *)result;
      }
    }
    *v12 = a2;
    *(_QWORD *)a1 = v12;
  }
  return result;
}
