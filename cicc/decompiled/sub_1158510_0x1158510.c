// Function: sub_1158510
// Address: 0x1158510
//
__int64 __fastcall sub_1158510(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // rdx
  _BYTE *v10; // rax
  int v11; // r14d
  unsigned int v12; // r15d
  _BYTE *v13; // rax

  v2 = *a2;
  if ( *a2 <= 0x1Cu )
  {
    v3 = 0;
    if ( v2 == 5 )
      return v3;
    return 0;
  }
  if ( (unsigned int)v2 - 48 > 1 && (unsigned __int8)(v2 - 55) > 1u )
    return 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  if ( v2 != 49 )
    return 0;
  v5 = *((_QWORD *)a2 - 8);
  if ( !v5 )
    return 0;
  **(_QWORD **)a1 = v5;
  v6 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v6 == 17 )
  {
    v3 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 8))(*(_QWORD *)(a1 + 16), v6 + 24);
  }
  else
  {
    v8 = *(_QWORD *)(v6 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v6 > 0x15u )
      return 0;
    v10 = sub_AD7630(v6, 0, v9);
    if ( !v10 || *v10 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v11 = *(_DWORD *)(v8 + 32);
        if ( v11 )
        {
          v3 = 0;
          v12 = 0;
          while ( 1 )
          {
            v13 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v6, v12);
            if ( !v13 )
              break;
            if ( *v13 != 13 )
            {
              if ( *v13 != 17 )
                break;
              v3 = (*(__int64 (__fastcall **)(_QWORD, _BYTE *))(a1 + 8))(*(_QWORD *)(a1 + 16), v13 + 24);
              if ( !(_BYTE)v3 )
                break;
            }
            if ( v11 == ++v12 )
              goto LABEL_13;
          }
        }
      }
      return 0;
    }
    v3 = (*(__int64 (__fastcall **)(_QWORD, _BYTE *))(a1 + 8))(*(_QWORD *)(a1 + 16), v10 + 24);
  }
LABEL_13:
  if ( !(_BYTE)v3 )
    return 0;
  v7 = *(__int64 **)(a1 + 24);
  if ( v7 )
    *v7 = v6;
  return v3;
}
