// Function: sub_10A7830
// Address: 0x10a7830
//
__int64 __fastcall sub_10A7830(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rax
  _BYTE *v12; // rdi
  char v13; // al
  __int64 v14; // rcx
  unsigned __int8 *v15; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v16; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6
    || *(_QWORD *)(v6 + 8)
    || *(_BYTE *)v5 != 44
    || (v12 = *(_BYTE **)(v5 - 64), *v12 > 0x15u)
    || (**a1 = v12, *v12 <= 0x15u) && (*v12 == 5 || (v16 = a3, v13 = sub_AD6CA0((__int64)v12), a3 = v16, v13))
    || (v14 = *(_QWORD *)(v5 - 32)) == 0 )
  {
    v7 = *((_QWORD *)a3 - 4);
  }
  else
  {
    *a1[2] = v14;
    v7 = *((_QWORD *)a3 - 4);
    if ( v7 )
    {
      *a1[3] = v7;
      return 1;
    }
  }
  v8 = *(_QWORD *)(v7 + 16);
  if ( v8 )
  {
    if ( !*(_QWORD *)(v8 + 8) && *(_BYTE *)v7 == 44 )
    {
      v9 = *(_BYTE **)(v7 - 64);
      if ( *v9 <= 0x15u )
      {
        **a1 = v9;
        if ( *v9 > 0x15u )
        {
LABEL_15:
          v10 = *(_QWORD *)(v7 - 32);
          if ( v10 )
          {
            *a1[2] = v10;
            v11 = *((_QWORD *)a3 - 8);
            if ( v11 )
            {
              *a1[3] = v11;
              return 1;
            }
          }
          return 0;
        }
        v15 = a3;
        if ( *v9 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v9) )
        {
          a3 = v15;
          goto LABEL_15;
        }
      }
    }
  }
  return 0;
}
