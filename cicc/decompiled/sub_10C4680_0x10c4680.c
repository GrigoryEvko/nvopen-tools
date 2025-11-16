// Function: sub_10C4680
// Address: 0x10c4680
//
__int64 __fastcall sub_10C4680(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  _BYTE *v8; // rax
  unsigned int v9; // eax
  int v10; // r14d
  unsigned int v11; // r15d
  _BYTE *v12; // rax
  unsigned int v13; // eax

  if ( *(_BYTE *)a2 == 17 )
  {
    LOBYTE(v2) = sub_B532C0(a2 + 24, *(_QWORD **)(a1 + 8), *(_DWORD *)a1);
    v3 = v2;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17;
    if ( (unsigned int)v6 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v8 = sub_AD7630(a2, 0, v6);
    if ( !v8 || *v8 != 17 )
    {
      if ( *(_BYTE *)(v5 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v5 + 32);
        if ( v10 )
        {
          v3 = 0;
          v11 = 0;
          while ( 1 )
          {
            v12 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a2, v11);
            if ( !v12 )
              break;
            if ( *v12 != 13 )
            {
              if ( *v12 != 17 )
                break;
              LOBYTE(v13) = sub_B532C0((__int64)(v12 + 24), *(_QWORD **)(a1 + 8), *(_DWORD *)a1);
              v3 = v13;
              if ( !(_BYTE)v13 )
                break;
            }
            if ( v10 == ++v11 )
              goto LABEL_3;
          }
        }
      }
      return 0;
    }
    LOBYTE(v9) = sub_B532C0((__int64)(v8 + 24), *(_QWORD **)(a1 + 8), *(_DWORD *)a1);
    v3 = v9;
  }
LABEL_3:
  if ( !(_BYTE)v3 )
    return 0;
  v4 = *(__int64 **)(a1 + 16);
  if ( v4 )
    *v4 = a2;
  return v3;
}
