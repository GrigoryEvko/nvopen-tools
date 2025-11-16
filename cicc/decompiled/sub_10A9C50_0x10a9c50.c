// Function: sub_10A9C50
// Address: 0x10a9c50
//
__int64 __fastcall sub_10A9C50(_QWORD **a1, int a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  _BYTE *v7; // r13
  unsigned __int8 *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  _BYTE *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v7 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v7 != 45
    || (result = sub_995E90(a1, *((_QWORD *)v7 - 8), (__int64)a3, a4, a5), !(_BYTE)result)
    || (a4 = *((_QWORD *)v7 - 4)) == 0 )
  {
    v8 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
    v9 = *v8;
LABEL_5:
    if ( (_BYTE)v9 == 45 )
    {
      result = sub_995E90(a1, *((_QWORD *)v8 - 8), v9, a4, a5);
      if ( (_BYTE)result )
      {
        v10 = *((_QWORD *)v8 - 4);
        if ( v10 )
        {
          *a1[1] = v10;
          v11 = (_BYTE *)*((_QWORD *)a3 - 8);
          if ( *v11 == 43 )
          {
            v12 = *((_QWORD *)v11 - 8);
            v13 = *((_QWORD *)v11 - 4);
            v14 = *a1[2];
            if ( v12 == v14 && v13 )
            {
LABEL_20:
              *a1[3] = v13;
              return result;
            }
            if ( v12 && v14 == v13 )
            {
              *a1[3] = v12;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  *a1[1] = a4;
  v8 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  v9 = *v8;
  if ( (_BYTE)v9 != 43 )
    goto LABEL_5;
  v13 = *((_QWORD *)v8 - 8);
  v15 = *((_QWORD *)v8 - 4);
  v16 = *a1[2];
  if ( v13 != v16 || !v15 )
  {
    if ( v13 && v15 == v16 )
      goto LABEL_20;
    return 0;
  }
  *a1[3] = v15;
  return result;
}
