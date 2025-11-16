// Function: sub_2DC40B0
// Address: 0x2dc40b0
//
__int64 __fastcall sub_2DC40B0(__int64 *a1, _BYTE *a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v3; // rax
  unsigned int v4; // edx
  __int64 v6; // rbx
  unsigned int v7; // r14d
  bool v8; // al
  __int64 *v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdx
  _BYTE *v12; // rax
  unsigned int v13; // r14d
  char v14; // r15
  unsigned int v15; // r14d
  __int64 v16; // rax
  unsigned int v17; // r15d
  int v18; // [rsp+Ch] [rbp-44h]

  v2 = 0;
  if ( *a2 != 82 )
    return v2;
  v3 = sub_B53900((__int64)a2);
  sub_B53630(v3, *a1);
  v2 = v4;
  if ( !(_BYTE)v4 || a1[1] != *((_QWORD *)a2 - 8) )
    return 0;
  v6 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v6 + 24) == 1;
    else
      v8 = v7 - 1 == (unsigned int)sub_C444A0(v6 + 24);
LABEL_9:
    if ( v8 )
      goto LABEL_10;
    return 0;
  }
  v10 = *(_QWORD *)(v6 + 8);
  v11 = (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17;
  if ( (unsigned int)v11 > 1 || *(_BYTE *)v6 > 0x15u )
    return 0;
  v12 = sub_AD7630(v6, 0, v11);
  if ( !v12 || *v12 != 17 )
  {
    if ( *(_BYTE *)(v10 + 8) == 17 )
    {
      v18 = *(_DWORD *)(v10 + 32);
      if ( v18 )
      {
        v14 = 0;
        v15 = 0;
        while ( 1 )
        {
          v16 = sub_AD69F0((unsigned __int8 *)v6, v15);
          if ( !v16 )
            break;
          if ( *(_BYTE *)v16 != 13 )
          {
            if ( *(_BYTE *)v16 != 17 )
              return 0;
            v17 = *(_DWORD *)(v16 + 32);
            if ( v17 <= 0x40 )
            {
              if ( *(_QWORD *)(v16 + 24) != 1 )
                return 0;
            }
            else if ( (unsigned int)sub_C444A0(v16 + 24) != v17 - 1 )
            {
              return 0;
            }
            v14 = v2;
          }
          if ( v18 == ++v15 )
          {
            if ( v14 )
              goto LABEL_10;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v13 = *((_DWORD *)v12 + 8);
  if ( v13 > 0x40 )
  {
    v8 = v13 - 1 == (unsigned int)sub_C444A0((__int64)(v12 + 24));
    goto LABEL_9;
  }
  if ( *((_QWORD *)v12 + 3) != 1 )
    return 0;
LABEL_10:
  v9 = (__int64 *)a1[2];
  if ( v9 )
    *v9 = v6;
  return v2;
}
