// Function: sub_33EBD60
// Address: 0x33ebd60
//
__int64 __fastcall sub_33EBD60(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  _QWORD *i; // rbx
  _QWORD *v6; // rdx
  unsigned __int64 v7; // rcx
  __int64 v8; // r8
  _QWORD *v9; // rbx
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+0h] [rbp-40h]

LABEL_1:
  result = *(unsigned int *)(a2 + 8);
  v3 = 8 * result - 8;
  while ( (_DWORD)result )
  {
    result = (unsigned int)(result - 1);
    v4 = *(_QWORD *)(*(_QWORD *)a2 + v3);
    *(_DWORD *)(a2 + 8) = result;
    v3 -= 8;
    if ( *(_DWORD *)(v4 + 24) )
    {
      for ( i = *(_QWORD **)(a1 + 768); i; i = (_QWORD *)i[1] )
        (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*i + 16LL))(i, v4, 0);
      sub_33EB970(a1, v4, v3);
      v9 = *(_QWORD **)(v4 + 40);
      v10 = (__int64)&v9[5 * *(unsigned int *)(v4 + 64)];
      if ( (_QWORD *)v10 != v9 )
      {
        do
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
            if ( !*(_QWORD *)(v11 + 56) )
              break;
            if ( v9 == (_QWORD *)v10 )
              goto LABEL_16;
          }
          v13 = *(unsigned int *)(a2 + 8);
          v7 = *(unsigned int *)(a2 + 12);
          if ( v13 + 1 > v7 )
          {
            v14 = v10;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 8u, v8, v10);
            v13 = *(unsigned int *)(a2 + 8);
            v10 = v14;
          }
          v6 = *(_QWORD **)a2;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = v11;
          ++*(_DWORD *)(a2 + 8);
        }
        while ( v9 != (_QWORD *)v10 );
      }
LABEL_16:
      sub_33CC1B0(a1, v4, (__int64)v6, v7, v8, v10);
      goto LABEL_1;
    }
  }
  return result;
}
