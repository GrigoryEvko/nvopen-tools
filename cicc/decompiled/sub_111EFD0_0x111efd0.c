// Function: sub_111EFD0
// Address: 0x111efd0
//
__int64 __fastcall sub_111EFD0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v8; // rax
  _BYTE *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _BYTE *v13; // rdi
  _BYTE *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx

  if ( !a2 )
    return 0;
  v3 = *(_QWORD *)(a2 - 64);
  v4 = *(_QWORD *)(v3 + 16);
  if ( v4 )
  {
    if ( !*(_QWORD *)(v4 + 8) && *(_BYTE *)v3 == 59 )
    {
      v8 = *(_QWORD *)(v3 - 64);
      if ( v8 )
      {
        *(_QWORD *)a1[1] = v8;
        v9 = *(_BYTE **)(v3 - 32);
        if ( *v9 <= 0x15u )
        {
          *(_QWORD *)a1[2] = v9;
          if ( *v9 > 0x15u || *v9 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v9) )
          {
            v5 = *(_QWORD *)(a2 - 32);
            if ( !v5 )
              goto LABEL_4;
            *(_QWORD *)a1[4] = v5;
            if ( *(_BYTE *)v5 > 0x15u || *(_BYTE *)v5 == 5 || (unsigned __int8)sub_AD6CA0(v5) )
            {
              if ( *a1 )
              {
                v10 = sub_B53900(a2);
                v11 = *a1;
                *(_DWORD *)v11 = v10;
                *(_BYTE *)(v11 + 4) = BYTE4(v10);
                return 1;
              }
              return 1;
            }
          }
        }
      }
    }
  }
  v5 = *(_QWORD *)(a2 - 32);
LABEL_4:
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 8) )
    return 0;
  if ( *(_BYTE *)v5 != 59 )
    return 0;
  v12 = *(_QWORD *)(v5 - 64);
  if ( !v12 )
    return 0;
  *(_QWORD *)a1[1] = v12;
  v13 = *(_BYTE **)(v5 - 32);
  if ( *v13 > 0x15u )
    return 0;
  *(_QWORD *)a1[2] = v13;
  if ( *v13 <= 0x15u && (*v13 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v13)) )
    return 0;
  v14 = *(_BYTE **)(a2 - 64);
  if ( !v14 )
    return 0;
  *(_QWORD *)a1[4] = v14;
  if ( *v14 <= 0x15u && *v14 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v14) )
    return 0;
  if ( *a1 )
  {
    v15 = sub_B53960(a2);
    v16 = *a1;
    *(_DWORD *)v16 = v15;
    *(_BYTE *)(v16 + 4) = BYTE4(v15);
  }
  return 1;
}
