// Function: sub_14A8A60
// Address: 0x14a8a60
//
__int64 __fastcall sub_14A8A60(__int64 a1, int *a2)
{
  _BYTE *v2; // rcx
  int v3; // edx
  unsigned __int64 v4; // rax
  __int64 v5; // r12
  _BYTE *v6; // rax
  _WORD *v8; // rdx

  v2 = *(_BYTE **)(a1 + 24);
  v3 = *a2;
  v4 = *(_QWORD *)(a1 + 16) - (_QWORD)v2;
  if ( *a2 )
  {
    if ( v3 != 4 )
    {
      if ( v3 == 2 )
      {
        if ( v4 <= 0xB )
        {
          v5 = sub_16E7EE0(a1, "notconstant<", 12);
        }
        else
        {
          v5 = a1;
          qmemcpy(v2, "notconstant<", 12);
          *(_QWORD *)(a1 + 24) += 12LL;
        }
      }
      else
      {
        if ( v3 == 3 )
        {
          if ( v4 <= 0xD )
          {
            v5 = sub_16E7EE0(a1, "constantrange<", 14);
          }
          else
          {
            v5 = a1;
            qmemcpy(v2, "constantrange<", 14);
            *(_QWORD *)(a1 + 24) += 14LL;
          }
          sub_16A95F0(a2 + 2, v5, 1);
          v8 = *(_WORD **)(v5 + 24);
          if ( *(_QWORD *)(v5 + 16) - (_QWORD)v8 <= 1u )
          {
            v5 = sub_16E7EE0(v5, ", ", 2);
          }
          else
          {
            *v8 = 8236;
            *(_QWORD *)(v5 + 24) += 2LL;
          }
          sub_16A95F0(a2 + 6, v5, 1);
          v6 = *(_BYTE **)(v5 + 24);
          if ( *(_BYTE **)(v5 + 16) != v6 )
            goto LABEL_8;
          return sub_16E7EE0(v5, ">", 1);
        }
        if ( v4 <= 8 )
        {
          v5 = sub_16E7EE0(a1, "constant<", 9);
        }
        else
        {
          v2[8] = 60;
          v5 = a1;
          *(_QWORD *)v2 = 0x746E6174736E6F63LL;
          *(_QWORD *)(a1 + 24) += 9LL;
        }
      }
      sub_155C2B0(*((_QWORD *)a2 + 1), v5, 0);
      v6 = *(_BYTE **)(v5 + 24);
      if ( *(_BYTE **)(v5 + 16) != v6 )
      {
LABEL_8:
        *v6 = 62;
        ++*(_QWORD *)(v5 + 24);
        return v5;
      }
      return sub_16E7EE0(v5, ">", 1);
    }
    if ( v4 > 0xA )
    {
      v5 = a1;
      qmemcpy(v2, "overdefined", 11);
      *(_QWORD *)(a1 + 24) += 11LL;
      return v5;
    }
    return sub_16E7EE0(a1, "overdefined", 11);
  }
  else
  {
    if ( v4 > 8 )
    {
      v2[8] = 100;
      v5 = a1;
      *(_QWORD *)v2 = 0x656E696665646E75LL;
      *(_QWORD *)(a1 + 24) += 9LL;
      return v5;
    }
    return sub_16E7EE0(a1, "undefined", 9);
  }
}
