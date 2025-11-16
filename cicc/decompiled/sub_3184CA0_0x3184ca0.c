// Function: sub_3184CA0
// Address: 0x3184ca0
//
// bad sp value at call has been detected, the output may be wrong!
__int64 __fastcall sub_3184CA0(__int64 *a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v5; // rdi
  char v6; // al
  __int64 v7; // r8
  char v9; // bl
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  unsigned __int8 v14; // al
  unsigned __int8 *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  _QWORD v18[6]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v19[10]; // [rsp+30h] [rbp-50h] BYREF

  v5 = *a1;
  v19[0] = a3;
  v18[0] = a2;
  v19[1] = -1;
  memset(&v19[2], 0, 32);
  v18[1] = -1;
  memset(&v18[2], 0, 32);
  v6 = sub_CF4E00(v5, (__int64)v18, (__int64)v19);
  if ( !v6 )
    return 0;
  LODWORD(v7) = 1;
  if ( (unsigned __int8)(v6 - 2) <= 1u )
    return (unsigned int)v7;
  v9 = sub_270F130(a2, (__int64)v18);
  v10 = sub_270F130(a3, (__int64)v18);
  if ( !v9 )
  {
    v11 = *a2;
    if ( !v10 || (_BYTE)v11 != 61 )
      goto LABEL_8;
    goto LABEL_17;
  }
  v16 = (__int64)a2;
  if ( *a3 == 61 )
    return sub_3183650(v16, (__int64)v18, v11, v12, v7, v13);
  v11 = *a2;
  if ( v10 )
  {
    v7 = 0;
    if ( (_BYTE)v11 != 61 )
      return (unsigned int)v7;
LABEL_17:
    v16 = (__int64)a3;
    return sub_3183650(v16, (__int64)v18, v11, v12, v7, v13);
  }
LABEL_8:
  if ( (_BYTE)v11 == 84 )
  {
    v17 = (__int64)a3;
  }
  else
  {
    v14 = *a3;
    if ( *a3 <= 0x1Cu )
    {
      LODWORD(v7) = 1;
      if ( (_BYTE)v11 != 86 )
        return (unsigned int)v7;
      goto LABEL_12;
    }
    if ( v14 != 84 )
    {
      if ( (_BYTE)v11 != 86 )
      {
        LODWORD(v7) = 1;
        if ( v14 != 86 )
          return (unsigned int)v7;
        v15 = a2;
        a2 = a3;
        return sub_3184A00((__int64)a1, (__int64)a2, v15);
      }
LABEL_12:
      v15 = a3;
      return sub_3184A00((__int64)a1, (__int64)a2, v15);
    }
    v17 = (__int64)a2;
    a2 = a3;
  }
  return sub_3184A90((__int64)a1, (__int64)a2, v17, v12, v7, v13);
}
