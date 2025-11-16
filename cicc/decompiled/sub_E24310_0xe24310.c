// Function: sub_E24310
// Address: 0xe24310
//
__int64 __fastcall sub_E24310(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  _BYTE *v4; // rdx
  __int64 v5; // rax
  _BYTE *v6; // rdx
  char v8; // cl

  v2 = *a2;
  v3 = 0;
  if ( !*a2 )
    return v3;
  v4 = (_BYTE *)a2[1];
  if ( *v4 == 69 )
  {
    --v2;
    a2[1] = (__int64)(v4 + 1);
    *a2 = v2;
    if ( !v2 )
      return 64;
    v8 = v4[1];
    v3 = 64;
    ++v4;
    if ( v8 != 73 )
      goto LABEL_4;
  }
  else if ( *v4 != 73 )
  {
LABEL_4:
    v5 = *a2;
    v6 = (_BYTE *)a2[1];
    goto LABEL_5;
  }
  v6 = v4 + 1;
  v5 = v2 - 1;
  v3 |= 0x20u;
  a2[1] = (__int64)v6;
  *a2 = v5;
  if ( !v5 )
    return v3;
LABEL_5:
  if ( *v6 != 70 )
    return v3;
  *a2 = v5 - 1;
  a2[1] = (__int64)(v6 + 1);
  return v3 | 0x10;
}
