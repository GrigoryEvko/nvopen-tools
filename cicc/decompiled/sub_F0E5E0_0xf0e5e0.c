// Function: sub_F0E5E0
// Address: 0xf0e5e0
//
__int64 __fastcall sub_F0E5E0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax
  __int64 v4; // rax
  int v5; // ecx
  __int64 v6; // rbx
  unsigned int v7; // r13d
  bool v8; // al
  __int64 v9; // r13
  __int64 v10; // rdx
  _BYTE *v11; // rax
  unsigned int v12; // ebx
  unsigned int v13; // ebx
  int v14; // r13d
  int v15; // edx
  unsigned __int8 *v16; // rax
  int v17; // r13d
  bool v18; // r14
  unsigned int v19; // r15d
  __int64 v20; // rax
  unsigned int v21; // r14d

  v2 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 44 )
  {
    v6 = *(_QWORD *)(a2 - 64);
    if ( *(_BYTE *)v6 == 17 )
    {
      v7 = *(_DWORD *)(v6 + 32);
      if ( v7 <= 0x40 )
        v8 = *(_QWORD *)(v6 + 24) == 0;
      else
        v8 = v7 == (unsigned int)sub_C444A0(v6 + 24);
    }
    else
    {
      v9 = *(_QWORD *)(v6 + 8);
      v10 = (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17;
      if ( (unsigned int)v10 > 1 || *(_BYTE *)v6 > 0x15u )
        return 0;
      v11 = sub_AD7630(v6, 0, v10);
      if ( !v11 || *v11 != 17 )
      {
        if ( *(_BYTE *)(v9 + 8) == 17 )
        {
          v17 = *(_DWORD *)(v9 + 32);
          if ( v17 )
          {
            v18 = 0;
            v19 = 0;
            while ( 1 )
            {
              v20 = sub_AD69F0((unsigned __int8 *)v6, v19);
              if ( !v20 )
                break;
              if ( *(_BYTE *)v20 != 13 )
              {
                if ( *(_BYTE *)v20 != 17 )
                  break;
                v21 = *(_DWORD *)(v20 + 32);
                v18 = v21 <= 0x40 ? *(_QWORD *)(v20 + 24) == 0 : v21 == (unsigned int)sub_C444A0(v20 + 24);
                if ( !v18 )
                  break;
              }
              if ( v17 == ++v19 )
              {
                if ( v18 )
                  goto LABEL_12;
                break;
              }
            }
          }
        }
LABEL_39:
        v2 = *(_BYTE *)a2;
        goto LABEL_2;
      }
      v12 = *((_DWORD *)v11 + 8);
      if ( v12 <= 0x40 )
        v8 = *((_QWORD *)v11 + 3) == 0;
      else
        v8 = v12 == (unsigned int)sub_C444A0((__int64)(v11 + 24));
    }
    if ( v8 )
    {
LABEL_12:
      result = *(_QWORD *)(a2 - 32);
      if ( result )
        return result;
    }
    goto LABEL_39;
  }
LABEL_2:
  if ( v2 == 17 )
    return sub_AD6890(a2, 0);
  if ( v2 == 16 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    if ( *(_BYTE *)(*(_QWORD *)(v4 + 24) + 8LL) == 12 )
      return sub_AD6890(a2, 0);
    goto LABEL_6;
  }
  if ( v2 != 11 )
  {
    if ( v2 <= 0x15u )
    {
      v4 = *(_QWORD *)(a2 + 8);
LABEL_6:
      v5 = *(unsigned __int8 *)(v4 + 8);
      if ( (unsigned __int8)(v5 - 17) <= 1u
        && *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL) == 12
        && sub_AD7630(a2, 0, (unsigned int)(v5 - 17)) )
      {
        return sub_AD6890(a2, 0);
      }
    }
    return 0;
  }
  v13 = 0;
  v14 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( !v14 )
    return sub_AD6890(a2, 0);
  while ( 1 )
  {
    v16 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)a2, v13);
    if ( !v16 )
      break;
    v15 = *v16;
    if ( (unsigned int)(v15 - 12) > 1 && (_BYTE)v15 != 17 )
      return 0;
    if ( v14 == ++v13 )
      return sub_AD6890(a2, 0);
  }
  return 0;
}
