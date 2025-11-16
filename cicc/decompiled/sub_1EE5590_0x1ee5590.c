// Function: sub_1EE5590
// Address: 0x1ee5590
//
__int64 __fastcall sub_1EE5590(
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
  int v14; // ecx
  unsigned int v15; // r13d
  unsigned int v16; // r12d
  __int64 v17; // r14
  int v18; // eax
  unsigned int v20; // [rsp+Ch] [rbp-64h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h]
  _DWORD *v24; // [rsp+28h] [rbp-48h]

  result = a7;
  *(_DWORD *)a4 = 0;
  if ( a2 )
  {
    for ( i = 0; a2 != i; ++i )
    {
      v15 = *(_DWORD *)(a3 + 4 * i);
      v16 = *(_DWORD *)(a1 + 4 * i);
      v17 = 4 * i;
      LOWORD(v14) = v15 - v16;
      if ( v15 == v16 )
        continue;
      result = *(unsigned int *)(v17 + *(_QWORD *)(a5 + 88));
      if ( !(_DWORD)result )
      {
        v20 = *(_DWORD *)(a3 + 4 * i) - v16;
        v21 = a8;
        v22 = a3;
        v23 = a1;
        v24 = (_DWORD *)(v17 + *(_QWORD *)(a5 + 88));
        v18 = sub_1ED7BB0(a5, i);
        LOWORD(v14) = v20;
        a8 = v21;
        a3 = v22;
        a1 = v23;
        *v24 = v18;
        result = *(unsigned int *)(*(_QWORD *)(a5 + 88) + 4 * i);
      }
      if ( a8 )
        result = (unsigned int)(*(_DWORD *)(v17 + a7) + result);
      if ( v16 < (unsigned int)result )
      {
        if ( v15 < (unsigned int)result )
          continue;
        v14 = v15 - result;
      }
      else
      {
        if ( v15 >= (unsigned int)result )
        {
LABEL_14:
          result = a4;
          *(_WORD *)a4 = i + 1;
          *(_WORD *)(a4 + 2) = v14;
          return result;
        }
        result = (unsigned int)result - v16;
        v14 = result;
      }
      if ( v14 )
        goto LABEL_14;
    }
  }
  return result;
}
