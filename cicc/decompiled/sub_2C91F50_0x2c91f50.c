// Function: sub_2C91F50
// Address: 0x2c91f50
//
__int64 __fastcall sub_2C91F50(char *a1, __int64 *a2)
{
  char v2; // al
  unsigned int v3; // r12d
  _QWORD *v5; // rax
  char *v6; // rdx
  char *v7; // rbx
  int v8; // eax
  char *v9; // rdx
  char *v10; // rbx
  int v11; // eax
  char *v12; // rdx
  char *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a1;
  if ( (unsigned __int8)*a1 > 0x1Cu )
  {
    if ( v2 != 57 )
    {
      if ( v2 == 58 )
      {
        if ( (a1[7] & 0x40) != 0 )
          v6 = (char *)*((_QWORD *)a1 - 1);
        else
          v6 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v3 = sub_2C91F50(*(_QWORD *)v6, &v17);
        if ( v3 )
        {
          v7 = (a1[7] & 0x40) != 0 ? (char *)*((_QWORD *)a1 - 1) : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
          v8 = sub_2C91F50(*((_QWORD *)v7 + 4), v18);
          if ( v8 )
          {
            if ( v3 == 1 && v8 == 1 )
            {
              *a2 = v18[0] | v17;
            }
            else
            {
              v3 = 2;
              *a2 = v18[0] + v17;
            }
            return v3;
          }
        }
      }
      else
      {
        v3 = 0;
        if ( v2 != 54 )
          return v3;
        v12 = (a1[7] & 0x40) != 0 ? (char *)*((_QWORD *)a1 - 1) : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v3 = sub_2C91F50(*(_QWORD *)v12, v18);
        if ( !v3 )
          return v3;
        if ( (a1[7] & 0x40) != 0 )
          v13 = (char *)*((_QWORD *)a1 - 1);
        else
          v13 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v14 = *((_QWORD *)v13 + 4);
        if ( *(_BYTE *)v14 == 17 )
        {
          v15 = *(_QWORD *)(v14 + 24);
          if ( *(_DWORD *)(v14 + 32) > 0x40u )
            v15 = *(_QWORD *)v15;
          if ( v15 <= 31 )
          {
            *a2 = v18[0] << v15;
            return v3;
          }
        }
      }
      return 0;
    }
    if ( (a1[7] & 0x40) != 0 )
      v9 = (char *)*((_QWORD *)a1 - 1);
    else
      v9 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v3 = sub_2C91F50(*(_QWORD *)v9, &v17);
    if ( (a1[7] & 0x40) != 0 )
      v10 = (char *)*((_QWORD *)a1 - 1);
    else
      v10 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v11 = sub_2C91F50(*((_QWORD *)v10 + 4), v18);
    if ( v3 == v11 )
    {
      if ( v3 == 1 )
      {
        *a2 = v18[0] & v17;
        return v3;
      }
      if ( v3 == 2 )
      {
        v16 = v18[0];
        if ( v17 >= v18[0] )
          v16 = v17;
        *a2 = v16;
        return v3;
      }
      return 0;
    }
    if ( v3 != 1 )
    {
      if ( v11 == 1 )
        goto LABEL_45;
      if ( v3 != 2 )
      {
        if ( v11 != 2 )
          return 0;
LABEL_45:
        v3 = 2;
        *a2 = v18[0];
        return v3;
      }
    }
    v3 = 2;
    *a2 = v17;
    return v3;
  }
  v3 = 0;
  if ( v2 != 17 )
    return v3;
  v5 = (_QWORD *)*((_QWORD *)a1 + 3);
  if ( *((_DWORD *)a1 + 8) > 0x40u )
    v5 = (_QWORD *)*v5;
  *a2 = (__int64)v5;
  return 1;
}
