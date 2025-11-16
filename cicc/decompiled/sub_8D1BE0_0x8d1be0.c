// Function: sub_8D1BE0
// Address: 0x8d1be0
//
__int64 __fastcall sub_8D1BE0(__int64 a1, _DWORD *a2)
{
  __int64 *v2; // rax
  char v3; // dl
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(__int64 **)(*(_QWORD *)(a1 + 168) + 168LL);
  v7[0] = v2;
  if ( v2 )
  {
    v3 = *((_BYTE *)v2 + 8);
    if ( v3 != 3 )
      goto LABEL_3;
    sub_72F220(v7);
    v2 = v7[0];
    if ( v7[0] )
    {
      v3 = *((_BYTE *)v7[0] + 8);
      while ( 1 )
      {
LABEL_3:
        if ( v3 == 2 )
        {
LABEL_8:
          v4 = *(_QWORD *)(v2[4] + 168);
          if ( v4 && (*(_BYTE *)(v4 + 266) & 1) != 0 )
            goto LABEL_10;
        }
        while ( 1 )
        {
          v2 = (__int64 *)*v2;
          v7[0] = v2;
          if ( !v2 )
            goto LABEL_11;
          v3 = *((_BYTE *)v2 + 8);
          if ( v3 != 3 )
            break;
          sub_72F220(v7);
          v2 = v7[0];
          if ( !v7[0] )
            goto LABEL_11;
          if ( *((_BYTE *)v7[0] + 8) == 2 )
            goto LABEL_8;
        }
      }
    }
  }
LABEL_11:
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
    return 0;
  if ( (*(_BYTE *)(a1 + 177) & 0x10) == 0 )
    return 0;
  v6 = sub_880FA0(a1);
  if ( !v6 || (*(_BYTE *)(*(_QWORD *)(v6 + 88) + 266LL) & 1) == 0 )
    return 0;
LABEL_10:
  *a2 = 1;
  return 1;
}
