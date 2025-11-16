// Function: sub_1F59570
// Address: 0x1f59570
//
char __fastcall sub_1F59570(__int64 a1)
{
  char v1; // dl
  unsigned int v3; // esi
  char v4; // cl
  __int64 v5; // rbx
  unsigned int v6; // eax
  _QWORD *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned int v10; // r14d
  char v11; // cl

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 11 )
  {
    v3 = *(_DWORD *)(a1 + 8) >> 8;
    if ( v3 == 32 )
      return 5;
    if ( v3 > 0x20 )
    {
      if ( v3 == 64 )
      {
        return 6;
      }
      else
      {
        if ( v3 != 128 )
          return sub_1F58CC0(*(_QWORD **)a1, v3);
        return 7;
      }
    }
    else if ( v3 == 8 )
    {
      return 3;
    }
    else
    {
      v4 = 4;
      if ( v3 != 16 )
      {
        v4 = 2;
        if ( v3 != 1 )
          return sub_1F58CC0(*(_QWORD **)a1, v3);
      }
    }
    return v4;
  }
  if ( v1 != 16 )
    return sub_1F59410(a1);
  v5 = *(_QWORD *)(a1 + 32);
  v6 = sub_1F59570(*(_QWORD *)(a1 + 24), 0);
  v7 = *(_QWORD **)a1;
  v9 = v8;
  v10 = v6;
  v11 = sub_1D15020(v6, v5);
  if ( !v11 )
    return sub_1F593D0(v7, v10, v9, v5);
  return v11;
}
