// Function: sub_C5F420
// Address: 0xc5f420
//
__int64 __fastcall sub_C5F420(__int64 a1, unsigned __int64 *a2, unsigned __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rax
  unsigned int v7; // r8d
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rbx
  char v11; // al

  if ( !a3 )
  {
    v10 = *a2;
    v11 = sub_C5EA20(a1, *a2, 2, 0, a5);
    v7 = 0;
    if ( v11 )
    {
      v7 = *(unsigned __int16 *)(*(_QWORD *)a1 + v10);
      if ( *(_BYTE *)(a1 + 16) != 1 )
        LOWORD(v7) = __ROL2__(v7, 8);
      *a2 += 2LL;
    }
    return v7;
  }
  v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
  if ( v6 )
  {
    v7 = 0;
LABEL_4:
    *a3 = v6 | 1;
    return v7;
  }
  *a3 = 0;
  v9 = *a2;
  if ( (unsigned __int8)sub_C5EA20(a1, *a2, 2, a3, a5) )
  {
    v7 = *(unsigned __int16 *)(*(_QWORD *)a1 + v9);
    if ( *(_BYTE *)(a1 + 16) != 1 )
      LOWORD(v7) = __ROL2__(v7, 8);
    *a2 += 2LL;
    v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    v7 = 0;
    v6 = *a3 & 0xFFFFFFFFFFFFFFFELL;
  }
  if ( v6 )
    goto LABEL_4;
  *a3 = 1;
  return v7;
}
