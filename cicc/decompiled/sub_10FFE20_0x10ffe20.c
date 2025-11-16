// Function: sub_10FFE20
// Address: 0x10ffe20
//
__int64 __fastcall sub_10FFE20(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 *v5; // rax
  __int64 v7; // r12
  _BYTE *v8; // rax
  unsigned int v9; // eax
  int v10; // r14d
  unsigned int v11; // r15d
  _BYTE *v12; // rax
  unsigned int v13; // eax

  if ( *(_BYTE *)a2 == 17 )
  {
    LOBYTE(v3) = sub_B532C0(a2 + 24, *(_QWORD **)(a1 + 8), *(_DWORD *)a1);
    v4 = v3;
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
      return 0;
    v8 = sub_AD7630(a2, 0, a3);
    if ( !v8 || *v8 != 17 )
    {
      if ( *(_BYTE *)(v7 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v7 + 32);
        if ( v10 )
        {
          v4 = 0;
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
              v4 = v13;
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
    v4 = v9;
  }
LABEL_3:
  if ( !(_BYTE)v4 )
    return 0;
  v5 = *(__int64 **)(a1 + 16);
  if ( v5 )
    *v5 = a2;
  return v4;
}
