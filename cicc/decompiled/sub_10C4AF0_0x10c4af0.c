// Function: sub_10C4AF0
// Address: 0x10c4af0
//
bool __fastcall sub_10C4AF0(_QWORD **a1, _BYTE *a2)
{
  bool result; // al
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx

  if ( *a2 != 58 )
    return 0;
  result = (a2[1] & 2) != 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  v3 = *((_QWORD *)a2 - 8);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4
    || *(_QWORD *)(v4 + 8)
    || *(_BYTE *)v3 != 46
    || (v9 = *(_QWORD *)(v3 - 64)) == 0
    || (**a1 = v9, (v10 = *(_QWORD *)(v3 - 32)) == 0) )
  {
    v5 = *((_QWORD *)a2 - 4);
LABEL_5:
    v6 = *(_QWORD *)(v5 + 16);
    if ( v6 )
    {
      if ( !*(_QWORD *)(v6 + 8) && *(_BYTE *)v5 == 46 )
      {
        v7 = *(_QWORD *)(v5 - 64);
        if ( v7 )
        {
          **a1 = v7;
          v8 = *(_QWORD *)(v5 - 32);
          if ( v8 )
          {
            *a1[1] = v8;
            return *a1[2] == *((_QWORD *)a2 - 8);
          }
        }
      }
    }
    return 0;
  }
  *a1[1] = v10;
  v5 = *((_QWORD *)a2 - 4);
  if ( *a1[2] != v5 )
    goto LABEL_5;
  return result;
}
