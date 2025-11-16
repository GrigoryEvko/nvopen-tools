// Function: sub_15403E0
// Address: 0x15403e0
//
char *__fastcall sub_15403E0(
        char *a1,
        __int64 *a2,
        unsigned __int8 (__fastcall *a3)(char *),
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  char *result; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  char *v11; // r15
  char *v12; // rbx
  _QWORD *v13; // r13
  int v14; // eax
  bool v15; // zf
  __int64 v16; // rax
  int v17; // eax
  char *v18; // rax
  __int64 v19; // rdx
  int v20; // ecx
  char *v21; // r13
  char *v22; // r15
  __int64 v23; // r12
  __int64 v24; // [rsp+0h] [rbp-50h]
  char *v25; // [rsp+8h] [rbp-48h]

  result = a1;
  if ( a4 != 1 )
  {
    v8 = a5;
    if ( a4 > a6 )
    {
      v24 = a4 / 2;
      v21 = &a1[16 * (a4 / 2)];
      v22 = v21;
      v25 = (char *)sub_15403E0(a1, v21, a3);
      v23 = a4 - v24;
      if ( v23 )
      {
        while ( a3(v22) )
        {
          v22 += 16;
          if ( !--v23 )
            return sub_153C900(v25, v21, v22);
        }
        v22 = (char *)sub_15403E0(v22, a2, a3);
      }
      return sub_153C900(v25, v21, v22);
    }
    else
    {
      v9 = *(_QWORD *)a1;
      v10 = a5 + 16;
      v11 = a1;
      v12 = a1 + 16;
      v13 = (_QWORD *)(a5 + 16);
      *(_QWORD *)a5 = *(_QWORD *)a1;
      *(_DWORD *)(a5 + 8) = *((_DWORD *)a1 + 2);
      if ( a1 + 16 == (char *)a2 )
      {
        *(_QWORD *)a1 = v9;
        *((_DWORD *)a1 + 2) = *(_DWORD *)(a5 + 8);
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v15 = a3(v12) == 0;
            v16 = *(_QWORD *)v12;
            if ( v15 )
              break;
            *(_QWORD *)v11 = v16;
            v14 = *((_DWORD *)v12 + 2);
            v11 += 16;
            v12 += 16;
            *((_DWORD *)v11 - 2) = v14;
            if ( a2 == (__int64 *)v12 )
              goto LABEL_8;
          }
          *v13 = v16;
          v17 = *((_DWORD *)v12 + 2);
          v13 += 2;
          v12 += 16;
          *((_DWORD *)v13 - 2) = v17;
        }
        while ( a2 != (__int64 *)v12 );
LABEL_8:
        v18 = v11;
        v19 = ((__int64)v13 - v8) >> 4;
        if ( (__int64)v13 - v8 > 0 )
        {
          while ( 1 )
          {
            v18 += 16;
            *((_QWORD *)v18 - 2) = *(_QWORD *)v8;
            v20 = *(_DWORD *)(v8 + 8);
            v8 = v10;
            *((_DWORD *)v18 - 2) = v20;
            if ( !--v19 )
              break;
            v10 += 16;
          }
        }
      }
      return v11;
    }
  }
  return result;
}
