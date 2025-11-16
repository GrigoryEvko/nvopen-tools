// Function: sub_10C27C0
// Address: 0x10c27c0
//
__int64 __fastcall sub_10C27C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  unsigned int v6; // r13d
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v10; // r14d
  bool v11; // al
  __int64 v12; // r14
  _BYTE *v13; // rax
  unsigned int v14; // ebx
  unsigned int v15; // r14d
  __int64 v16; // rax
  char v17; // [rsp+0h] [rbp-40h]
  int v18; // [rsp+0h] [rbp-40h]
  int v19; // [rsp+4h] [rbp-3Ch]

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  *(_QWORD *)a1[1] = v2;
  v3 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v3 > 0x15u )
    return 0;
  LOBYTE(v4) = sub_AC30F0(*(_QWORD *)(a2 - 32));
  v6 = v4;
  if ( (_BYTE)v4 )
    goto LABEL_5;
  if ( *(_BYTE *)v3 == 17 )
  {
    v10 = *(_DWORD *)(v3 + 32);
    if ( v10 <= 0x40 )
      v11 = *(_QWORD *)(v3 + 24) == 0;
    else
      v11 = v10 == (unsigned int)sub_C444A0(v3 + 24);
    goto LABEL_12;
  }
  v12 = *(_QWORD *)(v3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 )
    return v6;
  v13 = sub_AD7630(v3, 0, v5);
  if ( v13 && *v13 == 17 )
  {
    v14 = *((_DWORD *)v13 + 8);
    if ( v14 <= 0x40 )
    {
      if ( *((_QWORD *)v13 + 3) )
        return v6;
LABEL_5:
      v6 = 1;
      if ( *a1 )
      {
        v7 = sub_B53900(a2);
        v8 = *a1;
        *(_DWORD *)v8 = v7;
        *(_BYTE *)(v8 + 4) = BYTE4(v7);
      }
      return v6;
    }
    v11 = v14 == (unsigned int)sub_C444A0((__int64)(v13 + 24));
LABEL_12:
    if ( !v11 )
      return v6;
    goto LABEL_5;
  }
  if ( *(_BYTE *)(v12 + 8) == 17 )
  {
    v19 = *(_DWORD *)(v12 + 32);
    if ( v19 )
    {
      v17 = 0;
      v15 = 0;
      while ( 1 )
      {
        v16 = sub_AD69F0((unsigned __int8 *)v3, v15);
        if ( !v16 )
          break;
        if ( *(_BYTE *)v16 != 13 )
        {
          if ( *(_BYTE *)v16 != 17 )
            return v6;
          if ( *(_DWORD *)(v16 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v16 + 24) )
              return v6;
          }
          else
          {
            v18 = *(_DWORD *)(v16 + 32);
            if ( v18 != (unsigned int)sub_C444A0(v16 + 24) )
              return v6;
          }
          v17 = 1;
        }
        if ( v19 == ++v15 )
        {
          if ( v17 )
            goto LABEL_5;
          return v6;
        }
      }
    }
  }
  return v6;
}
