// Function: sub_7787E0
// Address: 0x7787e0
//
_BOOL8 __fastcall sub_7787E0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  _BOOL8 result; // rax
  __int64 v4; // r12
  __int64 v6; // rdx
  unsigned __int64 v8; // rbx
  char j; // dl
  unsigned int v10; // esi
  unsigned int v11; // ecx
  unsigned int i; // edx
  __int64 v13; // rax
  unsigned int k; // ecx
  __int64 v15; // rax
  int v16; // [rsp+Ch] [rbp-54h]
  unsigned __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v20[7]; // [rsp+28h] [rbp-38h] BYREF

  result = 0;
  v4 = a2[3];
  if ( v4 == a3[3] )
  {
    v6 = *a3;
    if ( *a2 != v6 )
    {
      if ( !*a2 || !v6 )
        return 0;
      v8 = *(_QWORD *)(v4 - 8);
      v16 = 1;
      j = *(_BYTE *)(v8 + 140);
LABEL_7:
      while ( j != 8 )
      {
LABEL_8:
        if ( (*(_BYTE *)(v4 - 9) & 2) != 0 && v16 )
        {
          v16 = 0;
          goto LABEL_11;
        }
        if ( (unsigned __int8)(j - 9) > 2u || *a2 == v4 || *a3 == v4 )
          return 1;
        sub_777100(a1, (__int64)a2, v4, v8, &v17, &v19);
        sub_777100(a1, (__int64)a3, v4, v8, &v18, v20);
        if ( v19 != v20[0] )
          return 0;
        if ( v19 )
        {
          v8 = *(_QWORD *)(v19 + 40);
          for ( i = qword_4F08388 & (v19 >> 3); ; i = qword_4F08388 & (i + 1) )
          {
            v13 = qword_4F08380 + 16LL * i;
            if ( v19 == *(_QWORD *)v13 )
              break;
            if ( !*(_QWORD *)v13 )
              goto LABEL_28;
          }
          v4 += *(unsigned int *)(v13 + 8);
LABEL_28:
          j = *(_BYTE *)(v8 + 140);
        }
        else
        {
          if ( v17 != v18 )
          {
            result = 0;
            if ( ((*(_BYTE *)(v18 + 88) ^ *(_BYTE *)(v17 + 88)) & 3) == 0 )
              return *(_BYTE *)(v8 + 140) != 11;
            return result;
          }
          v8 = *(_QWORD *)(v17 + 120);
          for ( j = *(_BYTE *)(v8 + 140); j == 12; j = *(_BYTE *)(v8 + 140) )
            v8 = *(_QWORD *)(v8 + 160);
          for ( k = qword_4F08388 & (v17 >> 3); ; k = qword_4F08388 & (k + 1) )
          {
            v15 = qword_4F08380 + 16LL * k;
            if ( v17 == *(_QWORD *)v15 )
              break;
            if ( !*(_QWORD *)v15 )
              goto LABEL_7;
          }
          v4 += *(unsigned int *)(v15 + 8);
        }
      }
      while ( 1 )
      {
        do
        {
          v8 = *(_QWORD *)(v8 + 160);
          j = *(_BYTE *)(v8 + 140);
        }
        while ( j == 12 );
LABEL_11:
        LODWORD(v20[0]) = 1;
        v10 = 16;
        if ( (unsigned __int8)(j - 2) > 1u )
          v10 = sub_7764B0(a1, v8, v20);
        v11 = ((unsigned int)*a2 - (unsigned int)v4) / v10;
        if ( v11 != ((unsigned int)*a3 - (unsigned int)v4) / v10 )
          break;
        j = *(_BYTE *)(v8 + 140);
        v4 += v11 * v10;
        if ( j != 8 )
          goto LABEL_8;
      }
    }
    return 1;
  }
  return result;
}
