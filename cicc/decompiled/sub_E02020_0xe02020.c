// Function: sub_E02020
// Address: 0xe02020
//
_QWORD *__fastcall sub_E02020(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  __int64 v7; // r15
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 v12; // r9
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rax
  const void *v16; // [rsp+0h] [rbp-50h]

  result = (_QWORD *)(a1 + 16);
  v16 = (const void *)(a1 + 16);
  if ( a3 )
  {
    v7 = a3;
    do
    {
      v9 = *(_QWORD *)(v7 + 24);
      v10 = sub_B43CB0(a5);
      result = (_QWORD *)sub_B43CB0(v9);
      if ( (_QWORD *)v10 == result )
      {
        result = (_QWORD *)sub_B19DB0(a6, a5, v9);
        if ( (_BYTE)result )
        {
          v13 = *(_BYTE *)v9;
          if ( *(_BYTE *)v9 == 78 )
          {
            result = (_QWORD *)sub_E02020(a1, a2, *(_QWORD *)(v9 + 16), a4, a5, a6);
          }
          else if ( v13 == 85 )
          {
            v14 = *(unsigned int *)(a1 + 8);
            if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, v16, v14 + 1, 0x10u, v11, v12);
              v14 = *(unsigned int *)(a1 + 8);
            }
            result = (_QWORD *)(*(_QWORD *)a1 + 16 * v14);
            result[1] = v9;
            *result = a4;
            ++*(_DWORD *)(a1 + 8);
          }
          else if ( v13 == 34 )
          {
            v15 = *(unsigned int *)(a1 + 8);
            if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, v16, v15 + 1, 0x10u, v11, v12);
              v15 = *(unsigned int *)(a1 + 8);
            }
            result = (_QWORD *)(*(_QWORD *)a1 + 16 * v15);
            result[1] = v9;
            *result = a4;
            ++*(_DWORD *)(a1 + 8);
          }
          else
          {
            result = a2;
            if ( a2 )
              *a2 = 1;
          }
        }
      }
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v7 );
  }
  return result;
}
