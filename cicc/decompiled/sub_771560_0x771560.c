// Function: sub_771560
// Address: 0x771560
//
__int64 __fastcall sub_771560(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int *a5,
        _DWORD *a6,
        _DWORD *a7)
{
  __int64 result; // rax
  unsigned int v11; // ecx
  unsigned __int8 v12; // al
  __int64 v14; // rax
  __int64 j; // rdx
  __int64 **k; // rcx
  int v17; // r8d
  __int64 *v18; // rax
  __int64 m; // rax
  __int64 i; // rax

  if ( *(_BYTE *)(a2 + 173) != 6 )
  {
    if ( !(unsigned int)sub_621000((__int16 *)(a2 + 176), 0, (__int16 *)&xmmword_4F08290, 0)
      && (*(_BYTE *)(a1 + 133) & 2) == 0 )
    {
      result = 0;
      v11 = 0;
      goto LABEL_4;
    }
    goto LABEL_3;
  }
  if ( !a4 )
  {
LABEL_3:
    result = 0;
    v11 = 0xFFFFFF;
    goto LABEL_4;
  }
  v12 = *(_BYTE *)(a2 + 176);
  if ( v12 == 4 )
  {
    for ( i = sub_8D46C0(*(_QWORD *)(a2 + 128)); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v11 = (unsigned int)*(_QWORD *)(i + 128) / a4;
    goto LABEL_25;
  }
  if ( v12 > 4u )
  {
    if ( v12 == 5 )
    {
      v11 = 0xFFFFFF;
LABEL_25:
      result = (unsigned int)*(_QWORD *)(a2 + 192) / a4;
      goto LABEL_4;
    }
LABEL_46:
    sub_721090();
  }
  if ( v12 == 1 )
  {
    k = *(__int64 ***)(a2 + 200);
    for ( j = *(_QWORD *)(*(_QWORD *)(a2 + 184) + 120LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    goto LABEL_12;
  }
  if ( (unsigned __int8)(v12 - 2) > 1u )
    goto LABEL_46;
  v14 = *(_QWORD *)(a2 + 184);
  if ( *(_BYTE *)(v14 + 173) == 2 )
  {
    v11 = (unsigned int)*(_QWORD *)(v14 + 176) / a4;
    goto LABEL_25;
  }
  j = *(_QWORD *)(v14 + 128);
  for ( k = *(__int64 ***)(a2 + 200); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
LABEL_12:
  v17 = 0;
  for ( result = 0; k; k = (__int64 **)*k )
  {
    if ( ((_BYTE)k[1] & 1) != 0 )
    {
      if ( *(_BYTE *)(j + 140) != 8 )
      {
        *a7 = 0;
        break;
      }
      for ( m = *(_QWORD *)(j + 160); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      if ( a3 == m )
      {
        result = *((unsigned int *)k + 4);
      }
      else
      {
        j = m;
        result = 0;
      }
    }
    else
    {
      v18 = k[2];
      if ( ((_BYTE)k[1] & 2) != 0 )
      {
        j = v18[5];
        result = 0;
      }
      else
      {
        for ( j = v18[15]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v17 = 1;
        result = 0;
      }
    }
  }
  v11 = 1;
  if ( *(_BYTE *)(j + 140) == 8 )
  {
    v11 = 0xFFFFFF;
    if ( (*(_WORD *)(j + 168) & 0x180) == 0 && (v17 || (*(_BYTE *)(j + 169) & 0x20) != 0 || *(_QWORD *)(j + 176)) )
      v11 = *(_QWORD *)(j + 176);
  }
LABEL_4:
  *a5 = v11;
  *a6 = result;
  return result;
}
