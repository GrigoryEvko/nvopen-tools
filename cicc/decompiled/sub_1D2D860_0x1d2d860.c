// Function: sub_1D2D860
// Address: 0x1d2d860
//
__int64 __fastcall sub_1D2D860(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  _QWORD *i; // rbx
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  _QWORD *v9; // rbx
  _QWORD *j; // r9
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // [rsp+0h] [rbp-40h]

LABEL_1:
  result = *(unsigned int *)(a2 + 8);
  v3 = 8 * result - 8;
  while ( (_DWORD)result )
  {
    result = (unsigned int)(result - 1);
    v4 = *(_QWORD *)(*(_QWORD *)a2 + v3);
    *(_DWORD *)(a2 + 8) = result;
    v3 -= 8;
    if ( *(_WORD *)(v4 + 24) )
    {
      for ( i = *(_QWORD **)(a1 + 664); i; i = (_QWORD *)i[1] )
        (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*i + 16LL))(i, v4, 0);
      sub_1D2D480(a1, v4, v3);
      v9 = *(_QWORD **)(v4 + 32);
      for ( j = &v9[5 * *(unsigned int *)(v4 + 56)]; v9 != j; ++*(_DWORD *)(a2 + 8) )
      {
        while ( 1 )
        {
          v11 = *v9;
          v9 += 5;
          if ( v11 )
          {
            v6 = (_QWORD *)*(v9 - 2);
            v12 = *(v9 - 1);
            *v6 = v12;
            if ( v12 )
            {
              v6 = (_QWORD *)*(v9 - 2);
              *(_QWORD *)(v12 + 24) = v6;
            }
          }
          *(v9 - 5) = 0;
          *((_DWORD *)v9 - 8) = 0;
          if ( *(_QWORD *)(a1 + 176) != v11 && !*(_QWORD *)(v11 + 48) )
            break;
          if ( v9 == j )
            goto LABEL_17;
        }
        v13 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v13 >= *(_DWORD *)(a2 + 12) )
        {
          v14 = j;
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v8, (int)j);
          v13 = *(unsigned int *)(a2 + 8);
          j = v14;
        }
        v6 = *(_QWORD **)a2;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = v11;
      }
LABEL_17:
      sub_1D17100(a1, v4, (__int64)v6, v7, v8, (int)j);
      goto LABEL_1;
    }
  }
  return result;
}
