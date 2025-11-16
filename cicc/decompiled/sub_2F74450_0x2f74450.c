// Function: sub_2F74450
// Address: 0x2f74450
//
__int64 __fastcall sub_2F74450(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 i; // rbx
  int v13; // r15d
  unsigned int v14; // r13d
  unsigned int v15; // r12d
  __int64 v16; // rdx
  int v17; // eax
  __int64 v19; // [rsp+10h] [rbp-60h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+30h] [rbp-40h]

  result = a7;
  *(_DWORD *)a4 = 0;
  if ( a2 )
  {
    for ( i = 0; a2 != i; ++i )
    {
      v14 = *(_DWORD *)(a3 + 4 * i);
      v15 = *(_DWORD *)(a1 + 4 * i);
      v16 = 4 * i;
      LOWORD(v13) = v14 - v15;
      if ( v14 == v15 )
        continue;
      result = *(unsigned int *)(*(_QWORD *)(a5 + 296) + 4 * i);
      if ( !(_DWORD)result )
      {
        v19 = a8;
        v20 = a3;
        v21 = a1;
        v22 = a5;
        v17 = sub_2F60A40(a5, i);
        a5 = v22;
        v16 = 4 * i;
        a8 = v19;
        a3 = v20;
        a1 = v21;
        *(_DWORD *)(*(_QWORD *)(v22 + 296) + 4 * i) = v17;
        result = *(unsigned int *)(*(_QWORD *)(v22 + 296) + 4 * i);
      }
      if ( a8 )
        result = (unsigned int)(*(_DWORD *)(v16 + a7) + result);
      if ( v15 < (unsigned int)result )
      {
        if ( v14 < (unsigned int)result )
          continue;
        v13 = v14 - result;
      }
      else
      {
        if ( v14 >= (unsigned int)result )
        {
LABEL_14:
          result = a4;
          *(_WORD *)a4 = i + 1;
          *(_WORD *)(a4 + 2) = v13;
          return result;
        }
        result = (unsigned int)result - v15;
        v13 = result;
      }
      if ( v13 )
        goto LABEL_14;
    }
  }
  return result;
}
