// Function: sub_85BA20
// Address: 0x85ba20
//
__int64 __fastcall sub_85BA20(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, _QWORD *a5, int *a6)
{
  __int64 *v9; // rax
  __int64 *v10; // r8
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 *v19; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v20[7]; // [rsp+18h] [rbp-38h] BYREF

  *a6 = 0;
  *a4 = 0;
  *a5 = 0;
  sub_89ED70(a1, a2, v20, &v19);
  v9 = v19;
  if ( !v19 )
  {
LABEL_20:
    v10 = 0;
LABEL_21:
    v11 = *a6;
    if ( *a6 )
      return (__int64)v10;
LABEL_11:
    v12 = unk_4F04C48;
    if ( unk_4F04C48 != -1 )
    {
      while ( 1 )
      {
        v13 = qword_4F04C68[0] + 776 * v12;
        if ( !v13 )
          break;
        if ( *(_BYTE *)(v13 + 4) == 9 )
        {
          v14 = **(_QWORD **)(v13 + 408);
          if ( v11 )
          {
            v15 = *(_QWORD *)(v13 + 376);
            if ( v15 )
              return sub_85BA20(v14, v15, a3, a4, a5, a6);
            v15 = *(_QWORD *)(v13 + 384);
            if ( v15 )
              return sub_85BA20(v14, v15, a3, a4, a5, a6);
            return (__int64)v10;
          }
          v11 = v14 == a1;
        }
        v12 = *(int *)(v13 + 552);
        if ( (_DWORD)v12 == -1 )
          return (__int64)v10;
      }
    }
    return (__int64)v10;
  }
  while ( *(_DWORD *)(*(_QWORD *)(v20[0] + 8LL) + 56LL) != *(_DWORD *)(a3 + 56) )
  {
    sub_89ED80(v20, &v19);
    v9 = v19;
    if ( !v19 )
      goto LABEL_20;
  }
  if ( *((_BYTE *)v9 + 8) == 3 )
  {
    v9 = (__int64 *)*v9;
    v19 = v9;
    if ( v9 && *((_BYTE *)v9 + 8) != 3 )
    {
      *a5 = v20[0];
      *a6 = 1;
      goto LABEL_7;
    }
    *a5 = v20[0];
    *a6 = 1;
    return 0;
  }
  *a5 = v20[0];
  *a6 = 1;
  if ( !v9 )
    return 0;
LABEL_7:
  v10 = v9;
  while ( (v9[3] & 8) != 0 )
  {
    ++*a4;
    v9 = (__int64 *)*v9;
    v19 = v9;
    if ( !v9 )
      goto LABEL_21;
  }
  v11 = *a6;
  if ( !*a6 )
    goto LABEL_11;
  return (__int64)v10;
}
