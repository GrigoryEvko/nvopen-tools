// Function: sub_28EECD0
// Address: 0x28eecd0
//
__int64 __fastcall sub_28EECD0(__int64 a1)
{
  unsigned __int8 *v1; // r12
  unsigned __int8 **v2; // rdx
  unsigned __int8 *v3; // r13
  __int64 v4; // r12
  unsigned int v5; // r13d
  bool v6; // al
  __int64 v8; // r13
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned int v11; // r12d
  unsigned __int8 *v12; // r13
  __int64 v13; // rax
  int v14; // r13d
  char v15; // r14
  unsigned int v16; // r15d
  __int64 v17; // rax
  unsigned int v18; // r14d

  if ( *(_BYTE *)a1 != 44 )
    goto LABEL_2;
  v4 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v4 == 17 )
  {
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
      v6 = *(_QWORD *)(v4 + 24) == 0;
    else
      v6 = v5 == (unsigned int)sub_C444A0(v4 + 24);
  }
  else
  {
    v8 = *(_QWORD *)(v4 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v4 > 0x15u )
      goto LABEL_2;
    v10 = sub_AD7630(*(_QWORD *)(a1 - 64), 0, v9);
    if ( !v10 || *v10 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v14 = *(_DWORD *)(v8 + 32);
        if ( v14 )
        {
          v15 = 0;
          v16 = 0;
          while ( 1 )
          {
            v17 = sub_AD69F0((unsigned __int8 *)v4, v16);
            if ( !v17 )
              break;
            if ( *(_BYTE *)v17 != 13 )
            {
              if ( *(_BYTE *)v17 != 17 )
                goto LABEL_2;
              v18 = *(_DWORD *)(v17 + 32);
              if ( v18 <= 0x40 )
              {
                if ( *(_QWORD *)(v17 + 24) )
                  goto LABEL_2;
              }
              else if ( v18 != (unsigned int)sub_C444A0(v17 + 24) )
              {
                goto LABEL_2;
              }
              v15 = 1;
            }
            if ( v14 == ++v16 )
            {
              if ( v15 )
                goto LABEL_12;
              goto LABEL_2;
            }
          }
        }
      }
      goto LABEL_2;
    }
    v11 = *((_DWORD *)v10 + 8);
    if ( v11 <= 0x40 )
    {
      if ( !*((_QWORD *)v10 + 3) )
        goto LABEL_12;
      goto LABEL_2;
    }
    v6 = v11 == (unsigned int)sub_C444A0((__int64)(v10 + 24));
  }
  if ( v6 )
    goto LABEL_12;
LABEL_2:
  LODWORD(v1) = sub_28EE9F0((unsigned __int8 *)a1);
  if ( (_BYTE)v1 )
  {
LABEL_12:
    LODWORD(v1) = 0;
    return (unsigned int)v1;
  }
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v2 = *(unsigned __int8 ***)(a1 - 8);
  else
    v2 = (unsigned __int8 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( (unsigned int)*v2[4] - 12 > 1 )
  {
    v3 = *v2;
    if ( !sub_28ED370(*v2, 13, 14) && !sub_28ED370(v3, 15, 16) )
    {
      v12 = *(unsigned __int8 **)(sub_986520(a1) + 32);
      if ( !sub_28ED370(v12, 13, 14) && !sub_28ED370(v12, 15, 16) )
      {
        v13 = *(_QWORD *)(a1 + 16);
        if ( *(_QWORD *)(v13 + 8) )
          return (unsigned int)v1;
        v1 = *(unsigned __int8 **)(v13 + 24);
        if ( !sub_28ED370(v1, 13, 14) )
        {
          LOBYTE(v1) = sub_28ED370(v1, 15, 16) != 0;
          return (unsigned int)v1;
        }
      }
    }
    LODWORD(v1) = 1;
  }
  return (unsigned int)v1;
}
