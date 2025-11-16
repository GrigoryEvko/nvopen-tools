// Function: sub_1230A50
// Address: 0x1230a50
//
__int64 __fastcall sub_1230A50(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int v4; // r15d
  __int64 v6; // rax
  __int16 v7; // dx
  __int64 v8; // r13
  __int64 v9; // r10
  int v10; // eax
  char v11; // al
  __int64 v12; // r10
  _BYTE *v13; // rsi
  char v14; // al
  const char *v15; // rax
  int v16; // eax
  __int64 v17; // [rsp+8h] [rbp-88h]
  __int64 v18; // [rsp+8h] [rbp-88h]
  unsigned __int64 v19; // [rsp+10h] [rbp-80h]
  __int64 v20; // [rsp+10h] [rbp-80h]
  __int64 *v22; // [rsp+20h] [rbp-70h] BYREF
  __int64 v23; // [rsp+28h] [rbp-68h] BYREF
  int v24[8]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v25; // [rsp+50h] [rbp-40h]

  v22 = 0;
  *(_QWORD *)v24 = "expected type";
  v25 = 259;
  v4 = sub_12190A0(a1, &v22, v24, 0);
  if ( !(_BYTE)v4 )
  {
    v25 = 257;
    v6 = sub_B49060((__int64)v22, 0, (__int64)v24, 0, 0);
    v7 = 0;
    v8 = v6;
    if ( *(_DWORD *)(a1 + 240) == 372 )
    {
      v16 = sub_1205200(a1 + 176);
      v7 = 1;
      *(_DWORD *)(a1 + 240) = v16;
    }
    v9 = a1 + 176;
    *(_WORD *)(v8 + 2) = v7 | *(_WORD *)(v8 + 2) & 0xFFFE;
    v10 = *(_DWORD *)(a1 + 240);
    if ( v10 == 373 )
      goto LABEL_11;
LABEL_6:
    if ( v10 == 374 )
    {
      v17 = v9;
      *(_DWORD *)(a1 + 240) = sub_1205200(v9);
      v19 = *(_QWORD *)(a1 + 232);
      v11 = sub_122FE20((__int64 **)a1, &v23, a3);
      v12 = v17;
      if ( !v11 )
      {
        v13 = (_BYTE *)v23;
        if ( *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) == 16 )
        {
          while ( *v13 <= 0x15u )
          {
            v20 = v12;
            sub_B49100(v8, (__int64)v13);
            v10 = *(_DWORD *)(a1 + 240);
            v9 = v20;
            if ( v10 != 373 )
              goto LABEL_6;
LABEL_11:
            v18 = v9;
            *(_DWORD *)(a1 + 240) = sub_1205200(v9);
            v19 = *(_QWORD *)(a1 + 232);
            v14 = sub_122FE20((__int64 **)a1, &v23, a3);
            v12 = v18;
            if ( v14 )
              goto LABEL_16;
            v13 = (_BYTE *)v23;
            if ( *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) == 16 )
            {
              HIBYTE(v25) = 1;
              v15 = "'catch' clause has an invalid type";
              goto LABEL_15;
            }
          }
          HIBYTE(v25) = 1;
          v15 = "clause argument must be a constant";
        }
        else
        {
          HIBYTE(v25) = 1;
          v15 = "'filter' clause has an invalid type";
        }
LABEL_15:
        *(_QWORD *)v24 = v15;
        LOBYTE(v25) = 3;
        sub_11FD800(v12, v19, (__int64)v24, 1);
      }
LABEL_16:
      v4 = 1;
      sub_B43C40(v8);
      sub_BD2DD0(v8);
    }
    else
    {
      *a2 = v8;
    }
  }
  return v4;
}
