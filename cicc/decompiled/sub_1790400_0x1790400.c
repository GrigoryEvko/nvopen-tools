// Function: sub_1790400
// Address: 0x1790400
//
__int64 __fastcall sub_1790400(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  unsigned __int8 v4; // al
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rsi

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 <= 0x17u )
    return 0;
  if ( v4 == 79 )
  {
    *a2 = *(_QWORD *)(a1 - 72);
    *a3 = *(_QWORD *)(a1 - 48);
    *a4 = *(_QWORD *)(a1 - 24);
    return a1;
  }
  if ( v4 != 61 )
  {
    if ( v4 == 62 )
    {
      v8 = *(__int64 **)(a1 - 24);
      *a2 = (__int64)v8;
      if ( sub_1642F90(*v8, 1) )
      {
        if ( sub_15FB730(*a2, 1, v9, v10) )
        {
          *a2 = sub_15FB7C0(*a2, 1, v11, v12);
          *a3 = sub_15A0680(*(_QWORD *)a1, 0, 0);
          *a4 = sub_15A0680(*(_QWORD *)a1, -1, 0);
          return a1;
        }
        v18 = -1;
        goto LABEL_15;
      }
    }
    return 0;
  }
  v13 = *(__int64 **)(a1 - 24);
  *a2 = (__int64)v13;
  if ( !sub_1642F90(*v13, 1) )
    return 0;
  if ( !sub_15FB730(*a2, 1, v14, v15) )
  {
    v18 = 1;
LABEL_15:
    *a3 = sub_15A0680(*(_QWORD *)a1, v18, 0);
    *a4 = sub_15A0680(*(_QWORD *)a1, 0, 0);
    return a1;
  }
  *a2 = sub_15FB7C0(*a2, 1, v16, v17);
  *a3 = sub_15A0680(*(_QWORD *)a1, 0, 0);
  *a4 = sub_15A0680(*(_QWORD *)a1, 1, 0);
  return a1;
}
