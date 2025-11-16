// Function: sub_C5F330
// Address: 0xc5f330
//
__int64 __fastcall sub_C5F330(_QWORD *a1, unsigned __int64 *a2, unsigned __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r14
  unsigned int v8; // r8d
  unsigned __int64 v10; // rbx
  char v11; // al
  unsigned int v12; // r8d

  if ( a3 )
  {
    v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
    if ( v6 )
    {
      v8 = 0;
    }
    else
    {
      *a3 = 0;
      v7 = *a2;
      if ( (unsigned __int8)sub_C5EA20((__int64)a1, *a2, 1, a3, a5) )
      {
        v8 = *(unsigned __int8 *)(*a1 + v7);
        ++*a2;
      }
      else
      {
        v8 = 0;
      }
      v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
      if ( !v6 )
      {
        *a3 = 1;
        return v8;
      }
    }
    *a3 = v6 | 1;
    return v8;
  }
  v10 = *a2;
  v11 = sub_C5EA20((__int64)a1, *a2, 1, 0, a5);
  v8 = 0;
  if ( !v11 )
    return v8;
  v12 = *(unsigned __int8 *)(*a1 + v10);
  ++*a2;
  return v12;
}
