// Function: sub_190D160
// Address: 0x190d160
//
char __fastcall sub_190D160(
        unsigned __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        __int64 a6,
        _QWORD *a7)
{
  unsigned __int64 i; // rax
  char v10; // dl
  unsigned __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // r15
  _BYTE *v14; // rsi
  char v16; // [rsp+16h] [rbp-6Ah]
  char v17; // [rsp+17h] [rbp-69h]
  __int64 v20; // [rsp+30h] [rbp-50h]
  unsigned __int64 v21; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v22; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v23; // [rsp+48h] [rbp-38h]

  v21 = a1;
  LOBYTE(i) = (unsigned __int8)sub_190CFA0(a7, &v21);
  v16 = v10;
  if ( v10 )
  {
    v11 = v21;
    if ( (*(_DWORD *)(v21 + 20) & 0xFFFFFFF) == 0 )
      goto LABEL_13;
    v17 = 0;
    v12 = 0;
    v20 = 24LL * ((*(_DWORD *)(v21 + 20) & 0xFFFFFFFu) - 1);
    if ( (*(_BYTE *)(v21 + 23) & 0x40) == 0 )
      goto LABEL_10;
LABEL_4:
    for ( i = *(_QWORD *)(v11 - 8); ; i = v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) )
    {
      v13 = *(_QWORD *)(i + v12);
      if ( *(_BYTE *)(v13 + 16) > 0x17u )
      {
        LOBYTE(i) = sub_15CC8F0(a3, *(_QWORD *)(v13 + 40), a2);
        if ( !(_BYTE)i )
        {
          v23 = v13;
          v22 = v21;
          sub_190D0C0(a4, &v22);
          v22 = v13;
          v23 = v21;
          sub_190D0C0(a5, &v22);
          sub_190D160(v13, a2, a3, (_DWORD)a4, (_DWORD)a5, a6, (__int64)a7);
          LOBYTE(i) = v16;
          v17 = v16;
        }
      }
      if ( v20 == v12 )
        break;
      v11 = v21;
      v12 += 24;
      if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
        goto LABEL_4;
LABEL_10:
      ;
    }
    if ( !v17 )
    {
LABEL_13:
      v14 = *(_BYTE **)(a6 + 8);
      if ( v14 == *(_BYTE **)(a6 + 16) )
      {
        LOBYTE(i) = (unsigned __int8)sub_170B610(a6, v14, &v21);
      }
      else
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = v21;
          v14 = *(_BYTE **)(a6 + 8);
        }
        *(_QWORD *)(a6 + 8) = v14 + 8;
        LOBYTE(i) = a6;
      }
    }
  }
  return i;
}
