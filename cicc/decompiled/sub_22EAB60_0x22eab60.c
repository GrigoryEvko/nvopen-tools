// Function: sub_22EAB60
// Address: 0x22eab60
//
__int64 __fastcall sub_22EAB60(char *a1, unsigned int a2, __int64 a3, char *a4, __int64 a5)
{
  char v5; // al
  unsigned __int8 v7; // dl
  __int64 *v8; // r15
  int v9; // eax
  __int64 v10; // rdi

  v5 = *a1;
  if ( !*a1 )
    return 0;
  v7 = *a4;
  if ( (unsigned __int8)*a4 <= 1u || v5 == 1 )
    return 0;
  if ( v5 != 2 || v7 != 2 )
  {
    if ( a2 - 32 <= 1 && (v5 == 3 && v7 == 2 || v7 == 3 && v5 == 2) )
    {
      if ( *((_QWORD *)a4 + 1) == *((_QWORD *)a1 + 1) )
      {
        v10 = a3;
        if ( a2 != 33 )
          return sub_AD6450(v10);
        return sub_AD6400(v10);
      }
    }
    else if ( (unsigned __int8)(v5 - 4) <= 1u && (unsigned __int8)(v7 - 4) <= 1u )
    {
      v8 = (__int64 *)(a4 + 8);
      if ( !sub_ABB410((__int64 *)a1 + 1, a2, (__int64 *)a4 + 1) )
      {
        v9 = sub_B52870(a2);
        if ( sub_ABB410((__int64 *)a1 + 1, v9, v8) )
        {
          v10 = a3;
          return sub_AD6450(v10);
        }
        return 0;
      }
      v10 = a3;
      return sub_AD6400(v10);
    }
    return 0;
  }
  return sub_9719A0(a2, *((_BYTE **)a1 + 1), *((_QWORD *)a4 + 1), a5, 0, 0);
}
