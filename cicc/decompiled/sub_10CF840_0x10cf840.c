// Function: sub_10CF840
// Address: 0x10cf840
//
__int64 __fastcall sub_10CF840(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v10; // rdi
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rax

  v10 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v10 = **(_QWORD **)(v10 + 16);
  if ( !sub_BCAC40(v10, 1) )
    return 0;
  v11 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 82 )
  {
LABEL_5:
    if ( v11 != 83 || *(_BYTE *)a3 != 83 || (result = sub_10C2DD0((__int64)a1, a2, a3, a5, a6)) == 0 )
    {
LABEL_6:
      v12 = *(_QWORD *)(a2 + 16);
      if ( v12 )
      {
        if ( !*(_QWORD *)(v12 + 8) )
        {
          v14 = *(_QWORD *)(a3 + 16);
          if ( v14 )
          {
            if ( !*(_QWORD *)(v14 + 8) )
              return sub_10C1A40((__int64)a1, (unsigned __int8 *)a2, (unsigned __int8 *)a3, a5);
          }
        }
      }
      return 0;
    }
    return result;
  }
  if ( *(_BYTE *)a3 != 82 )
    goto LABEL_6;
  result = (__int64)sub_10CC690(a1, a2, a3, a4, a5, a6);
  if ( !result )
  {
    v11 = *(_BYTE *)a2;
    goto LABEL_5;
  }
  return result;
}
