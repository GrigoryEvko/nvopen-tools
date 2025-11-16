// Function: sub_10008C0
// Address: 0x10008c0
//
__int64 __fastcall sub_10008C0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rdi
  __int64 v10; // r13
  int v11; // r14d
  __int64 v12; // rdx
  unsigned int v13; // r15d
  bool v14; // al
  int v15; // eax
  unsigned int v16; // edx
  int v17; // r15d
  bool v18; // al
  __int64 v19; // rdx
  _BYTE *v20; // rax
  __int64 v21; // r15
  _BYTE *v22; // rax
  unsigned __int8 *v23; // rdx
  unsigned int v24; // r15d
  unsigned int v25; // r15d
  __int64 v26; // rax
  char v27; // [rsp-50h] [rbp-50h]
  int v28; // [rsp-50h] [rbp-50h]
  int v29; // [rsp-4Ch] [rbp-4Ch]
  __int64 v30; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v31; // [rsp-48h] [rbp-48h]

  if ( !a1 )
    return 0;
  v4 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v4 != 85 )
    return 0;
  v7 = *(_QWORD *)(v4 - 32);
  if ( !v7 )
    return 0;
  if ( *(_BYTE *)v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v4 + 80) )
    return 0;
  if ( *(_DWORD *)(v7 + 36) != 66 )
    return 0;
  v8 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
  if ( !v8 )
    return 0;
  v9 = *(_QWORD *)(a1 - 32);
  v10 = v9 + 24;
  if ( *(_BYTE *)v9 != 17 )
  {
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 > 1 )
      return 0;
    if ( *(_BYTE *)v9 > 0x15u )
      return 0;
    v20 = sub_AD7630(v9, 0, v19);
    if ( !v20 || *v20 != 17 )
      return 0;
    v10 = (__int64)(v20 + 24);
  }
  v11 = sub_B53900(a1);
  if ( !a2 || v8 != *(_QWORD *)(a2 - 64) )
    return 0;
  v12 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v12 == 17 )
  {
    v13 = *(_DWORD *)(v12 + 32);
    v14 = v13 <= 0x40 ? *(_QWORD *)(v12 + 24) == 0 : v13 == (unsigned int)sub_C444A0(v12 + 24);
  }
  else
  {
    v21 = *(_QWORD *)(v12 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 || *(_BYTE *)v12 > 0x15u )
      return 0;
    v30 = *(_QWORD *)(a2 - 32);
    v22 = sub_AD7630(v30, 0, v12);
    v23 = (unsigned __int8 *)v30;
    if ( !v22 || *v22 != 17 )
    {
      if ( *(_BYTE *)(v21 + 8) == 17 )
      {
        v29 = *(_DWORD *)(v21 + 32);
        if ( v29 )
        {
          v27 = 0;
          v25 = 0;
          while ( 1 )
          {
            v31 = v23;
            v26 = sub_AD69F0(v23, v25);
            if ( !v26 )
              break;
            v23 = v31;
            if ( *(_BYTE *)v26 != 13 )
            {
              if ( *(_BYTE *)v26 != 17 )
                return 0;
              if ( *(_DWORD *)(v26 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v26 + 24) )
                  return 0;
                v27 = 1;
              }
              else
              {
                v28 = *(_DWORD *)(v26 + 32);
                if ( v28 != (unsigned int)sub_C444A0(v26 + 24) )
                  return 0;
                v27 = 1;
                v23 = v31;
              }
            }
            if ( v29 == ++v25 )
            {
              if ( v27 )
                goto LABEL_18;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v24 = *((_DWORD *)v22 + 8);
    v14 = v24 <= 0x40 ? *((_QWORD *)v22 + 3) == 0 : v24 == (unsigned int)sub_C444A0((__int64)(v22 + 24));
  }
  if ( !v14 )
    return 0;
LABEL_18:
  v15 = sub_B53900(a2);
  v16 = *(_DWORD *)(v10 + 8);
  v17 = v15;
  if ( v16 <= 0x40 )
    v18 = *(_QWORD *)v10 == 0;
  else
    v18 = v16 == (unsigned int)sub_C444A0(v10);
  if ( v18 )
    return 0;
  result = 0;
  if ( a3 )
  {
    if ( v11 == 33 && v17 == 32 )
      return a2;
  }
  else if ( v11 == 32 && v17 == 33 )
  {
    return a2;
  }
  return result;
}
