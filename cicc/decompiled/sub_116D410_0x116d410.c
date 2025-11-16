// Function: sub_116D410
// Address: 0x116d410
//
__int64 __fastcall sub_116D410(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _QWORD *v7; // r12
  _QWORD *v8; // rax
  char v10; // dl
  __int64 v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // r13

  v6 = a3;
  v7 = (_QWORD *)a1;
  if ( !*(_BYTE *)(a3 + 28) )
    goto LABEL_8;
  v8 = *(_QWORD **)(a3 + 8);
  a4 = *(unsigned int *)(a3 + 20);
  a3 = (__int64)&v8[a4];
  if ( v8 != (_QWORD *)a3 )
  {
    while ( a1 != *v8 )
    {
      if ( (_QWORD *)a3 == ++v8 )
        goto LABEL_7;
    }
    return 1;
  }
LABEL_7:
  if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
  {
    *(_DWORD *)(v6 + 20) = a4 + 1;
    *(_QWORD *)a3 = a1;
    ++*(_QWORD *)v6;
  }
  else
  {
LABEL_8:
    sub_C8CC70(v6, a1, a3, a4, a5, a6);
    if ( !v10 )
      return 1;
  }
  if ( *(_DWORD *)(v6 + 20) - *(_DWORD *)(v6 + 24) != 16 )
  {
    v11 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    {
      v12 = *(_QWORD **)(a1 - 8);
      v7 = &v12[v11];
    }
    else
    {
      v12 = (_QWORD *)(a1 - v11 * 8);
    }
    if ( v12 != v7 )
    {
      while ( 1 )
      {
        v13 = *v12;
        if ( *(_BYTE *)*v12 == 84 )
        {
          if ( !(unsigned __int8)sub_116D410(*v12, a2, v6) )
          {
            if ( *a2 )
              return 0;
            *a2 = v13;
          }
        }
        else if ( *a2 != v13 )
        {
          return 0;
        }
        v12 += 4;
        if ( v7 == v12 )
          return 1;
      }
    }
    return 1;
  }
  return 0;
}
