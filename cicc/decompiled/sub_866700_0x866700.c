// Function: sub_866700
// Address: 0x866700
//
__int64 *__fastcall sub_866700(_DWORD *a1, int a2, __int64 a3, int a4)
{
  __int64 v6; // rdx
  __int64 **v7; // rbx
  __int64 *v8; // rax
  _DWORD *v9; // rdx
  __int64 *v10; // r12
  __int64 *result; // rax
  unsigned __int8 v12; // di
  char v13; // al
  __int64 *v14; // rax
  unsigned __int8 v15; // di
  char v16; // al

  if ( !qword_4F04C18 || *((_BYTE *)qword_4F04C18 + 42) || (v6 = qword_4F04C18[2]) == 0 )
  {
    if ( !a3 )
      return 0;
LABEL_13:
    v12 = 0;
    v13 = *(_BYTE *)(*(_QWORD *)(a3 + 8) + 80LL);
    if ( v13 != 3 )
      v12 = (v13 != 2) + 1;
    v14 = sub_725090(v12);
    v10 = v14;
    if ( !a2 )
      return v10;
    sub_895F30(v14);
    return v10;
  }
  v7 = *(__int64 ***)(v6 + 8);
  v8 = *(__int64 **)(qword_4F04C18[1] + 24LL);
  if ( !v8 )
  {
LABEL_18:
    if ( !a4 )
      return 0;
    goto LABEL_13;
  }
  while ( 1 )
  {
    if ( !*((_DWORD *)v8 + 8) )
    {
      v9 = (_DWORD *)v8[8];
      if ( v9[1] == a1[1] && *v9 == *a1 )
        break;
    }
    v8 = (__int64 *)*v8;
    v7 = (__int64 **)*v7;
    if ( !v8 )
      goto LABEL_18;
  }
  v10 = v7[10];
  if ( !*((_BYTE *)qword_4F04C18 + 41) )
  {
    if ( v10 || !a4 )
      return v10;
    goto LABEL_13;
  }
  if ( v10 && *((_BYTE *)v10 + 8) != 3 )
    return v10;
  v15 = 0;
  v16 = *(_BYTE *)(v8[1] + 80);
  if ( v16 != 3 )
    v15 = (v16 != 2) + 1;
  result = sub_725090(v15);
  *((_BYTE *)result + 24) |= 8u;
  *result = *v7[11];
  *v7[11] = (__int64)result;
  v7[11] = result;
  v7[10] = result;
  return result;
}
