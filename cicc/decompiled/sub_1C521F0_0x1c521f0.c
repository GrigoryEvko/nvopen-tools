// Function: sub_1C521F0
// Address: 0x1c521f0
//
__int64 __fastcall sub_1C521F0(__int64 a1, __int64 *a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r12d
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rbx
  int v8; // eax
  _QWORD *v9; // rax
  __int64 v10; // rbx
  int v11; // eax
  _QWORD *v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 > 0x17u )
  {
    if ( v2 != 50 )
    {
      if ( v2 == 51 )
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v6 = *(_QWORD **)(a1 - 8);
        else
          v6 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        v3 = sub_1C521F0(*v6, &v17);
        if ( v3 )
        {
          v7 = (*(_BYTE *)(a1 + 23) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          v8 = sub_1C521F0(*(_QWORD *)(v7 + 24), v18);
          if ( v8 )
          {
            if ( v3 == 1 && v8 == 1 )
            {
              *a2 = v18[0] | v17;
            }
            else
            {
              v3 = 2;
              *a2 = v18[0] + v17;
            }
            return v3;
          }
        }
      }
      else
      {
        v3 = 0;
        if ( v2 != 47 )
          return v3;
        v12 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
            ? *(_QWORD **)(a1 - 8)
            : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        v3 = sub_1C521F0(*v12, v18);
        if ( !v3 )
          return v3;
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v13 = *(_QWORD *)(a1 - 8);
        else
          v13 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v14 = *(_QWORD *)(v13 + 24);
        if ( *(_BYTE *)(v14 + 16) == 13 )
        {
          v15 = *(_QWORD *)(v14 + 24);
          if ( *(_DWORD *)(v14 + 32) > 0x40u )
            v15 = *(_QWORD *)v15;
          if ( v15 <= 31 )
          {
            *a2 = v18[0] << v15;
            return v3;
          }
        }
      }
      return 0;
    }
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v9 = *(_QWORD **)(a1 - 8);
    else
      v9 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v3 = sub_1C521F0(*v9, &v17);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v10 = *(_QWORD *)(a1 - 8);
    else
      v10 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v11 = sub_1C521F0(*(_QWORD *)(v10 + 24), v18);
    if ( v3 == v11 )
    {
      if ( v3 == 1 )
      {
        *a2 = v18[0] & v17;
        return v3;
      }
      if ( v3 == 2 )
      {
        v16 = v18[0];
        if ( v17 >= v18[0] )
          v16 = v17;
        *a2 = v16;
        return v3;
      }
      return 0;
    }
    if ( v3 != 1 )
    {
      if ( v11 == 1 )
        goto LABEL_45;
      if ( v3 != 2 )
      {
        if ( v11 != 2 )
          return 0;
LABEL_45:
        v3 = 2;
        *a2 = v18[0];
        return v3;
      }
    }
    v3 = 2;
    *a2 = v17;
    return v3;
  }
  v3 = 0;
  if ( v2 != 13 )
    return v3;
  v5 = *(_QWORD **)(a1 + 24);
  if ( *(_DWORD *)(a1 + 32) > 0x40u )
    v5 = (_QWORD *)*v5;
  *a2 = (__int64)v5;
  return 1;
}
