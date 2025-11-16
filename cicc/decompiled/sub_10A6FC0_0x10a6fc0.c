// Function: sub_10A6FC0
// Address: 0x10a6fc0
//
__int64 __fastcall sub_10A6FC0(_QWORD **a1, __int64 a2)
{
  unsigned __int8 **v2; // r13
  char v4; // al
  unsigned __int8 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int8 v8; // al
  unsigned int v9; // r13d
  bool v10; // al
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  _BYTE *v14; // rax
  unsigned int v15; // r13d
  int v16; // r13d
  char v17; // r14
  unsigned int v18; // r15d
  __int64 v19; // rax
  unsigned int v20; // r14d

  if ( !a2 )
    return 0;
  v2 = *(unsigned __int8 ***)(a2 - 64);
  if ( *(_BYTE *)v2 != 42 )
    return 0;
  v4 = sub_996420(a1, 30, *(v2 - 8));
  v5 = *(v2 - 4);
  if ( v4 && v5 )
  {
    *a1[2] = v5;
    v7 = *(_QWORD *)(a2 - 32);
    v8 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 != 17 )
      goto LABEL_17;
LABEL_9:
    v9 = *(_DWORD *)(v7 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v7 + 24) == 1;
    else
      v10 = v9 - 1 == (unsigned int)sub_C444A0(v7 + 24);
LABEL_11:
    if ( !v10 )
      return 0;
    goto LABEL_12;
  }
  if ( !(unsigned __int8)sub_996420(a1, 30, v5) )
    return 0;
  v6 = (__int64)*(v2 - 8);
  if ( !v6 )
    return 0;
  *a1[2] = v6;
  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(_BYTE *)v7;
  if ( *(_BYTE *)v7 == 17 )
    goto LABEL_9;
LABEL_17:
  v12 = *(_QWORD *)(v7 + 8);
  v13 = (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17;
  if ( (unsigned int)v13 > 1 || v8 > 0x15u )
    return 0;
  v14 = sub_AD7630(v7, 0, v13);
  if ( !v14 || *v14 != 17 )
  {
    if ( *(_BYTE *)(v12 + 8) == 17 )
    {
      v16 = *(_DWORD *)(v12 + 32);
      if ( v16 )
      {
        v17 = 0;
        v18 = 0;
        while ( 1 )
        {
          v19 = sub_AD69F0((unsigned __int8 *)v7, v18);
          if ( !v19 )
            break;
          if ( *(_BYTE *)v19 != 13 )
          {
            if ( *(_BYTE *)v19 != 17 )
              return 0;
            v20 = *(_DWORD *)(v19 + 32);
            if ( v20 <= 0x40 )
            {
              if ( *(_QWORD *)(v19 + 24) != 1 )
                return 0;
            }
            else if ( (unsigned int)sub_C444A0(v19 + 24) != v20 - 1 )
            {
              return 0;
            }
            v17 = 1;
          }
          if ( v16 == ++v18 )
          {
            if ( v17 )
              goto LABEL_12;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v15 = *((_DWORD *)v14 + 8);
  if ( v15 > 0x40 )
  {
    v10 = v15 - 1 == (unsigned int)sub_C444A0((__int64)(v14 + 24));
    goto LABEL_11;
  }
  if ( *((_QWORD *)v14 + 3) != 1 )
    return 0;
LABEL_12:
  v11 = a1[3];
  if ( v11 )
    *v11 = v7;
  return 1;
}
