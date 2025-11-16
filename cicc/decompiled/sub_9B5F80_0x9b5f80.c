// Function: sub_9B5F80
// Address: 0x9b5f80
//
__int64 __fastcall sub_9B5F80(_BYTE *a1, unsigned __int8 *a2, __int64 a3, int a4, __m128i *a5)
{
  unsigned int v6; // r12d
  unsigned __int8 *v8; // rdi
  char v9; // al
  __int64 v10; // rdx
  __m128i *v11; // r8
  unsigned __int8 *v12; // rdi
  unsigned int v13; // ecx
  char v14; // al

  if ( *a1 != 86 )
    return 0;
  v6 = a4 + 1;
  v8 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
  if ( *a2 == 86 && *((_QWORD *)a1 - 12) == *((_QWORD *)a2 - 12) )
  {
    v14 = sub_9B5220(v8, *((unsigned __int8 **)a2 - 8), a3, v6, a5);
    v10 = a3;
    v11 = a5;
    if ( v14 )
    {
      a2 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      v12 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
      v13 = v6;
      return sub_9B5220(v12, a2, v10, v13, v11);
    }
    return 0;
  }
  v9 = sub_9B5220(v8, a2, a3, v6, a5);
  v10 = a3;
  v11 = a5;
  if ( !v9 )
    return 0;
  v12 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  v13 = v6;
  return sub_9B5220(v12, a2, v10, v13, v11);
}
