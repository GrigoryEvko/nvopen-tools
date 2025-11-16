// Function: sub_3364B70
// Address: 0x3364b70
//
bool __fastcall sub_3364B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, bool *a6)
{
  char v9; // al
  bool result; // al
  int v11; // ecx
  bool v12; // r8
  int v13; // edx
  bool v14; // di
  bool v15; // r12
  int v16; // esi
  int v17; // r9d
  bool v18; // r9
  bool v19; // si
  bool v20; // cl
  int v21; // ebx
  _BYTE *v22; // rax
  _BYTE *v23; // rdx
  __int64 v25; // [rsp+18h] [rbp-B8h] BYREF
  _QWORD v26[8]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v27[14]; // [rsp+60h] [rbp-70h] BYREF

  sub_33644B0((__int64)v26, a1, a5);
  if ( !v26[0] )
    return 0;
  sub_33644B0((__int64)v27, a3, a5);
  if ( !v27[0] )
    return 0;
  v9 = sub_3364290((__int64)v26, (__int64)v27, a5, &v25);
  if ( !v9 )
  {
    v11 = *(_DWORD *)(v26[0] + 24LL);
    v12 = v11 == 39;
    v13 = *(_DWORD *)(v27[0] + 24LL);
    v14 = v11 == 39 || v11 == 15;
    if ( v14 )
    {
      v15 = v13 == 15 || v13 == 39;
      if ( v15 )
      {
        v16 = *(_DWORD *)(v26[0] + 96LL);
        v17 = *(_DWORD *)(v27[0] + 96LL);
        if ( v17 != v16 )
        {
          if ( v16 >= 0 )
            goto LABEL_12;
          v21 = -*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a5 + 40) + 48LL) + 32LL);
          if ( v17 <= v16 )
            v16 = *(_DWORD *)(v27[0] + 96LL);
          if ( v21 > v16 || v17 >= 0 )
            goto LABEL_12;
        }
      }
      if ( v11 == 15 )
      {
        v18 = 0;
        v12 = 1;
        v19 = (unsigned int)(v13 - 13) <= 1 || (unsigned int)(v13 - 37) <= 1;
        v9 = 0;
        if ( v13 != 17 )
        {
LABEL_30:
          if ( v13 != 41 )
          {
            if ( v14 )
            {
              if ( !v19 && !v15 )
                return 0;
LABEL_34:
              if ( !(v19 ^ v18 | v15 ^ v12) && !v9 )
              {
                if ( !v18 )
                  return 0;
                if ( !v19 )
                  return 0;
                v22 = *(_BYTE **)(v26[0] + 96LL);
                v23 = *(_BYTE **)(v27[0] + 96LL);
                if ( v23 == v22 || *v22 == 1 || *v23 == 1 )
                  return 0;
              }
              goto LABEL_12;
            }
            if ( v9 )
              goto LABEL_18;
            return 0;
          }
          if ( v14 )
          {
            v9 ^= 1u;
            goto LABEL_34;
          }
          if ( !v9 )
            return 0;
          goto LABEL_45;
        }
LABEL_56:
        v9 ^= 1u;
        v12 = v14;
        goto LABEL_34;
      }
    }
    else
    {
      v15 = v13 == 39 || v13 == 15;
    }
    v18 = (unsigned int)(v11 - 13) <= 1 || (unsigned int)(v11 - 37) <= 1;
    v19 = (unsigned int)(v13 - 13) <= 1 || (unsigned int)(v13 - 37) <= 1;
    if ( v11 != 17 )
    {
      v9 = v11 == 41;
      v14 = v18 || v12;
      if ( v13 != 17 )
        goto LABEL_30;
      if ( v14 )
      {
        v14 = v11 == 39;
        goto LABEL_56;
      }
      if ( v11 != 41 )
        return 0;
LABEL_45:
      if ( !v19 && !v15 )
        return 0;
      goto LABEL_12;
    }
    if ( v13 != 17 )
    {
      v20 = v18 || v12;
      if ( v13 == 41 )
      {
        if ( v20 )
          goto LABEL_34;
      }
      else if ( !v20 )
      {
LABEL_18:
        if ( !v15 && !v19 )
          return 0;
        goto LABEL_12;
      }
      goto LABEL_45;
    }
    if ( v19 || v15 )
      goto LABEL_34;
    if ( !v18 )
      return 0;
LABEL_12:
    *a6 = 0;
    return 1;
  }
  if ( v25 < 0 )
  {
    result = a4 != 0xBFFFFFFFFFFFFFFELL && a4 != -1;
    if ( !result || (a4 & 0x4000000000000000LL) != 0 )
      return 0;
    *a6 = v25 + (a4 & 0x3FFFFFFFFFFFFFFFLL) > 0;
  }
  else
  {
    result = a2 != 0xBFFFFFFFFFFFFFFELL && a2 != -1;
    if ( !result || (a2 & 0x4000000000000000LL) != 0 )
      return 0;
    *a6 = v25 < (a2 & 0x3FFFFFFFFFFFFFFFLL);
  }
  return result;
}
