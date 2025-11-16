// Function: sub_36F3F80
// Address: 0x36f3f80
//
__int64 __fastcall sub_36F3F80(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 *v4; // rax
  _QWORD *v5; // r15
  unsigned __int64 v6; // r13
  int v7; // eax
  _QWORD *v8; // rdi
  __int64 v9; // r9
  __int64 v11; // [rsp+10h] [rbp-80h]
  _QWORD **v12; // [rsp+18h] [rbp-78h]
  unsigned __int8 v13; // [rsp+27h] [rbp-69h]
  _QWORD *v14; // [rsp+28h] [rbp-68h]
  _BYTE v15[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v16; // [rsp+50h] [rbp-40h]

  v13 = sub_BB98D0((_QWORD *)a1, a2);
  if ( !v13 && (!*(_BYTE *)(a1 + 169) || *(_BYTE *)(a1 + 170)) )
  {
    v3 = (_QWORD *)sub_B2BE50(a2);
    v4 = (__int64 *)sub_BCB120(v3);
    v12 = (_QWORD **)sub_BCF640(v4, 0);
    v11 = sub_B41A60(v12, (__int64)"exit;", 5, (__int64)byte_3F871B3, 0, 1, 0, 0, 0);
    v14 = *(_QWORD **)(a2 + 80);
    if ( v14 != (_QWORD *)(a2 + 72) )
    {
      while ( 1 )
      {
        if ( !v14 )
          BUG();
        v5 = (_QWORD *)v14[4];
        if ( v5 != v14 + 3 )
          break;
LABEL_20:
        v14 = (_QWORD *)v14[1];
        if ( (_QWORD *)(a2 + 72) == v14 )
          return v13;
      }
      while ( 1 )
      {
        if ( !v5 )
          BUG();
        if ( *((_BYTE *)v5 - 24) != 36 )
          goto LABEL_19;
        if ( v5 != *(_QWORD **)(v5[2] + 56LL) )
        {
          v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v6 )
          {
            if ( *(_BYTE *)(v6 - 24) == 85 )
            {
              v7 = sub_B49240(v6 - 24);
              if ( (v7 == 354 || v7 == 361)
                && !(unsigned __int8)sub_A747A0((_QWORD *)(v6 + 48), "trap-func-name", 0xEu)
                && !(unsigned __int8)sub_B49590(v6 - 24, "trap-func-name", 0xEu) )
              {
                goto LABEL_19;
              }
              if ( *(_BYTE *)(a1 + 170)
                && ((unsigned __int8)sub_A73ED0((_QWORD *)(v6 + 48), 36) || (unsigned __int8)sub_B49560(v6 - 24, 36)) )
              {
LABEL_16:
                v16 = 257;
                v8 = sub_BD2C40(88, 1u);
                if ( v8 )
                  sub_B4A410((__int64)v8, (__int64)v12, v11, (__int64)v15, 1u, v9, (__int64)v5, 0);
                v13 = 1;
                goto LABEL_19;
              }
            }
          }
        }
        if ( !*(_BYTE *)(a1 + 169) )
          goto LABEL_16;
LABEL_19:
        v5 = (_QWORD *)v5[1];
        if ( v14 + 3 == v5 )
          goto LABEL_20;
      }
    }
  }
  return 0;
}
