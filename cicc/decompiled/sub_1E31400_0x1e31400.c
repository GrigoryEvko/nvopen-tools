// Function: sub_1E31400
// Address: 0x1e31400
//
unsigned __int64 __fastcall sub_1E31400(
        char *a1,
        int a2,
        unsigned __int8 a3,
        unsigned __int8 a4,
        char a5,
        char a6,
        unsigned __int8 a7,
        unsigned __int8 a8)
{
  unsigned __int8 v9; // r8
  __int64 v13; // rdx
  char v14; // si
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 result; // rax
  int v19; // [rsp+0h] [rbp-40h]
  unsigned __int8 v20; // [rsp+7h] [rbp-39h]

  v9 = a6 | a5;
  v13 = *((_QWORD *)a1 + 2);
  v14 = *a1;
  if ( v13 && (v15 = *(_QWORD *)(v13 + 24)) != 0 && (v16 = *(_QWORD *)(v15 + 56)) != 0 )
  {
    v17 = *(_QWORD *)(v16 + 40);
    if ( v17 && !v14 )
    {
      v19 = a2;
      v20 = v9;
      sub_1E69A50(v17, a1);
      *((_QWORD *)a1 + 3) = 0;
      *((_DWORD *)a1 + 2) = v19;
      *(_QWORD *)a1 = *(_QWORD *)a1 & 0xFFFFFFF00FF00000LL
                    | (((unsigned __int64)a8 << 35)
                     | ((unsigned __int64)a7 << 32)
                     | ((unsigned __int64)v20 << 30)
                     | ((unsigned __int64)a3 << 28)
                     | ((unsigned __int64)a4 << 29))
                    & 0xFF00FFFFFLL;
      return sub_1E699D0(v17, a1);
    }
    result = *(_QWORD *)a1 & 0xFFFFFFF00FF00000LL;
    *((_DWORD *)a1 + 2) = a2;
    *((_QWORD *)a1 + 3) = 0;
    *(_QWORD *)a1 = result
                  | (((unsigned __int64)a3 << 28)
                   | ((unsigned __int64)a4 << 29)
                   | ((unsigned __int64)v9 << 30)
                   | ((unsigned __int64)a7 << 32)
                   | ((unsigned __int64)a8 << 35))
                  & 0xFF00FFFFFLL;
    if ( !v14 )
      goto LABEL_10;
  }
  else
  {
    v17 = 0;
    *((_DWORD *)a1 + 2) = a2;
    *((_QWORD *)a1 + 3) = 0;
    result = *(_QWORD *)a1 & 0xFFFFFFF00FF00000LL
           | (((unsigned __int64)a8 << 35)
            | ((unsigned __int64)a7 << 32)
            | ((unsigned __int64)v9 << 30)
            | ((unsigned __int64)a3 << 28)
            | ((unsigned __int64)a4 << 29))
           & 0xFF00FFFFFLL;
    *(_QWORD *)a1 = result;
    if ( !v14 )
      return result;
  }
  *((_WORD *)a1 + 1) &= 0xF00Fu;
LABEL_10:
  if ( v17 )
    return sub_1E699D0(v17, a1);
  return result;
}
