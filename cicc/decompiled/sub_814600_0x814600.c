// Function: sub_814600
// Address: 0x814600
//
__int64 __fastcall sub_814600(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 k; // rbx
  __int64 m; // rbx
  __int64 result; // rax
  __int64 n; // rbx
  char v10; // al
  __int64 ii; // rbx
  __int64 v12; // rdx
  __int64 i; // rbx
  char v14; // al
  __int64 j; // rbx
  char v16; // al
  __int64 v17; // [rsp+0h] [rbp-70h] BYREF
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  int v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]
  _BOOL4 v24; // [rsp+38h] [rbp-38h]
  char v25; // [rsp+3Ch] [rbp-34h]
  __int64 v26; // [rsp+40h] [rbp-30h]

  sub_814A10(*(_QWORD *)(a1 + 104));
  if ( *(_BYTE *)(a1 + 28) )
    goto LABEL_2;
  a2 = (__int64 *)sub_814A10;
  sub_76C540(qword_4F07288, sub_814A10);
  for ( i = *(_QWORD *)(a1 + 96); i; i = *(_QWORD *)(i + 120) )
  {
    v14 = *(_BYTE *)(i + 89);
    if ( (v14 & 4) != 0 && *(_BYTE *)(i + 173) != 12 && (v14 & 8) == 0 )
      sub_8134A0(i, (__int64)sub_814A10, v12, v3, v4, v5);
  }
  v2 = (unsigned int)(unk_4D043C8 | unk_4D04170);
  if ( !(unk_4D043C8 | unk_4D04170) )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_2;
    if ( unk_4F07778 <= 201102 )
    {
      v3 = dword_4F07774;
      if ( !dword_4F07774 )
        goto LABEL_2;
    }
    v2 = dword_4F077BC;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 <= 0xC34Fu )
          goto LABEL_2;
        goto LABEL_35;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_2;
    }
    if ( qword_4F077A0 <= 0x76BFu )
      goto LABEL_2;
  }
LABEL_35:
  for ( j = *(_QWORD *)(a1 + 112); j; j = *(_QWORD *)(j + 112) )
  {
    if ( (*(_BYTE *)(j + 89) & 8) == 0 && (unsigned int)sub_80D1B0(j) )
    {
      v16 = *(_BYTE *)(j + 170);
      v20 = 0;
      v24 = (v16 & 0x20) != 0;
      LOBYTE(v21) = 0;
      v22 = 0;
      v23 = 0;
      v25 = 0;
      v26 = 0;
      sub_809110(j, a2, v2, v3, v4, v5, 0, 0, 0);
      sub_823800(qword_4F18BE0);
      v17 += 2;
      sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
      sub_812EE0(j, &v17);
      a2 = 0;
      sub_80B290(j, 0, (__int64)&v17);
    }
  }
LABEL_2:
  for ( k = *(_QWORD *)(a1 + 168); k; k = *(_QWORD *)(k + 112) )
  {
    if ( (*(_BYTE *)(k + 124) & 1) == 0 )
    {
      v17 = 0;
      v18 = 0;
      v19 = 0;
      v20 = 0;
      LOBYTE(v21) = 0;
      v22 = 0;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      v26 = 0;
      if ( *(_QWORD *)(k + 8) || (a2 = &v17, sub_80B070(k, (__int64)&v17), !(_DWORD)v23) )
        sub_814600(*(_QWORD *)(k + 128), a2, v2, v3, v4, v5, v17, v18, v19, v20, v21, v22);
    }
  }
  for ( m = *(_QWORD *)(a1 + 144); m; m = *(_QWORD *)(m + 112) )
  {
    while ( (*(_DWORD *)(m + 192) & 0x8000400) != 0 )
    {
      m = *(_QWORD *)(m + 112);
      if ( !m )
        goto LABEL_10;
    }
    a2 = 0;
    sub_814390(m, 0);
  }
LABEL_10:
  result = *(unsigned __int8 *)(a1 + 28);
  if ( (_BYTE)result == 6 || (_BYTE)result == 3 )
  {
    for ( n = *(_QWORD *)(a1 + 112); n; n = *(_QWORD *)(n + 112) )
    {
      while ( 1 )
      {
        if ( (*(_BYTE *)(n + 89) & 8) == 0 )
        {
          result = sub_80D1B0(n);
          if ( (_DWORD)result )
            break;
        }
        n = *(_QWORD *)(n + 112);
        if ( !n )
          goto LABEL_21;
      }
      v10 = *(_BYTE *)(n + 170);
      v20 = 0;
      v24 = (v10 & 0x20) != 0;
      LOBYTE(v21) = 0;
      v22 = 0;
      v23 = 0;
      v25 = 0;
      v26 = 0;
      sub_809110(n, a2, v2, v3, v4, v5, 0, 0, 0);
      sub_823800(qword_4F18BE0);
      v17 += 2;
      sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
      sub_812EE0(n, &v17);
      a2 = 0;
      result = (__int64)sub_80B290(n, 0, (__int64)&v17);
    }
LABEL_21:
    for ( ii = *(_QWORD *)(a1 + 96); ii; ii = *(_QWORD *)(ii + 120) )
    {
      if ( (*(_BYTE *)(ii + 89) & 8) == 0 )
        result = sub_8134A0(ii, (__int64)a2, v2, v3, v4, v5);
    }
  }
  return result;
}
