// Function: sub_622780
// Address: 0x622780
//
__int64 __fastcall sub_622780(__m128i *a1, int a2, __int64 a3, unsigned __int8 a4, unsigned int *a5)
{
  unsigned int v7; // ebx
  __int64 v8; // rax
  char *v9; // rax
  __int64 result; // rax
  __int64 v11; // rax

  v7 = a4;
  *a5 = 0;
  if ( a2 )
  {
    v8 = sub_620EE0(a1, 1, a5);
    if ( *a5 )
    {
LABEL_3:
      v9 = sub_622500(a1, a2);
      return sub_70AFD0(v7, v9, a3, a5);
    }
    sub_70B680(v7, v8, a3, a5);
  }
  else
  {
    v11 = sub_620F30(a1->m128i_i16, 0, a5);
    if ( *a5 )
      goto LABEL_3;
    sub_70B6D0(v7, v11, a3, a5);
  }
  result = *a5;
  if ( (_DWORD)result )
    goto LABEL_3;
  return result;
}
