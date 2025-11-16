// Function: sub_7D4400
// Address: 0x7d4400
//
__int64 __fastcall sub_7D4400(
        __int64 *a1,
        _BYTE *a2,
        __int64 *a3,
        unsigned int a4,
        __int64 a5,
        __int64 *a6,
        _DWORD *a7,
        int a8)
{
  __int64 v11; // r15
  __int64 v12; // r11
  __int64 v13; // rax
  char v14; // al
  _BYTE *v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+8h] [rbp-48h]

  if ( !a2 )
    goto LABEL_18;
  if ( (a2[124] & 1) == 0 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
    if ( v11 )
      goto LABEL_4;
LABEL_18:
    if ( !a3 )
      return 0;
    v11 = 0;
    goto LABEL_5;
  }
  v11 = *(_QWORD *)(*(_QWORD *)sub_735B70((__int64)a2) + 96LL);
  if ( !v11 )
    goto LABEL_18;
LABEL_4:
  *(_BYTE *)(v11 + 200) |= 1u;
  if ( !a3 )
  {
    v12 = 0;
LABEL_23:
    *(_BYTE *)(v11 + 200) &= ~1u;
    return v12;
  }
LABEL_5:
  v12 = 0;
  do
  {
    v14 = *((_BYTE *)a3 + 40);
    if ( (v14 & 1) != 0 && (!a8 || (v14 & 0x20) != 0) )
    {
      v15 = (_BYTE *)a3[3];
      v16 = (__int64)v15;
      if ( (v15[124] & 1) != 0 )
      {
        v23 = v12;
        v17 = sub_735B70((__int64)v15);
        v12 = v23;
        v16 = v17;
        if ( (*(_BYTE *)(v17 + 124) & 1) != 0 )
        {
          v22 = v23;
          v24 = v17;
          v17 = sub_735B70(v17);
          v16 = v24;
          v12 = v22;
        }
        v15 = (_BYTE *)v17;
      }
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v15 + 96LL) + 200LL) & 1) == 0 )
      {
        v13 = sub_7D3C60(a1, v16, a4, a5, (int)a6, a8, a7);
        v12 = v13;
        if ( v13 )
        {
          if ( (*(_BYTE *)(v13 + 82) & 8) != 0 )
          {
            *a6 = v13;
          }
          else
          {
            v19 = *a6;
            if ( !*a6 )
            {
              v25 = v13;
              v21 = sub_7CF9D0(*a1, a4, 1, a5);
              v12 = v25;
              *a6 = v21;
              v19 = v21;
            }
            v20 = sub_7D09E0(v19, v12, (__int64)a1, 1u, a5, a4, a7);
            *a6 = v20;
            v12 = v20;
          }
        }
        else
        {
          v12 = *a6;
        }
      }
    }
    a3 = (__int64 *)*a3;
  }
  while ( a3 );
  if ( v11 )
    goto LABEL_23;
  return v12;
}
