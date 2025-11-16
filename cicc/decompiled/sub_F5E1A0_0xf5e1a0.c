// Function: sub_F5E1A0
// Address: 0xf5e1a0
//
__int64 __fastcall sub_F5E1A0(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // r14
  __int64 v4; // rbx
  _QWORD *v5; // r12
  char v6; // al
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // rsi
  __int64 v10; // rax
  const char *v11; // rax
  __int64 *v12; // rdx
  __int16 v13; // dx
  __int64 v14; // rsi
  unsigned __int8 v15; // cl
  char v16; // dl
  __int64 v17; // rax
  _BYTE *v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+18h] [rbp-68h] BYREF
  const char *v20; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v21; // [rsp+28h] [rbp-58h]
  const char *v22; // [rsp+30h] [rbp-50h]
  __int16 v23; // [rsp+40h] [rbp-40h]

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x15u )
    return sub_AD63D0(a1);
  v20 = 0;
  v21 = &v19;
  if ( v1 != 59 )
    goto LABEL_4;
  v8 = sub_995B10((_QWORD **)&v20, *(_QWORD *)(a1 - 64));
  v9 = *(_QWORD *)(a1 - 32);
  if ( v8 && v9 )
  {
    *v21 = v9;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10((_QWORD **)&v20, v9) || (v10 = *(_QWORD *)(a1 - 64)) == 0 )
    {
      v1 = *(_BYTE *)a1;
LABEL_4:
      if ( v1 <= 0x1Cu )
      {
        if ( v1 == 22 )
        {
          v3 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 80LL);
          v18 = 0;
          if ( v3 )
            v3 -= 24;
        }
        else
        {
          v18 = 0;
          v3 = 0;
        }
      }
      else
      {
        v18 = (_BYTE *)a1;
        v3 = *(_QWORD *)(a1 + 40);
      }
      v4 = *(_QWORD *)(a1 + 16);
      if ( v4 )
      {
        while ( 1 )
        {
          v5 = *(_QWORD **)(v4 + 24);
          if ( *(_BYTE *)v5 > 0x1Cu && v3 == v5[5] )
          {
            v20 = 0;
            v21 = (__int64 *)a1;
            if ( *(_BYTE *)v5 == 59 )
            {
              v6 = sub_995B10((_QWORD **)&v20, *(v5 - 8));
              v7 = *(v5 - 4);
              if ( v6 )
              {
                if ( (__int64 *)v7 == v21 )
                  break;
              }
              if ( (unsigned __int8)sub_995B10((_QWORD **)&v20, v7) && (__int64 *)*(v5 - 8) == v21 )
                break;
            }
          }
          v4 = *(_QWORD *)(v4 + 8);
          if ( !v4 )
            goto LABEL_24;
        }
      }
      else
      {
LABEL_24:
        v11 = sub_BD5D20(a1);
        v23 = 773;
        v20 = v11;
        v21 = v12;
        v22 = ".inv";
        v5 = (_QWORD *)sub_B50640(a1, (__int64)&v20, 0, 0);
        if ( !v18 || *v18 == 84 )
        {
          v14 = sub_AA5190(v3);
          if ( v14 )
          {
            v15 = v13;
            v16 = HIBYTE(v13);
          }
          else
          {
            v16 = 0;
            v15 = 0;
          }
          v17 = v15;
          BYTE1(v17) = v16;
          sub_B44220(v5, v14, v17);
        }
        else
        {
          sub_B43E90((__int64)v5, (__int64)(v18 + 24));
        }
      }
      return (__int64)v5;
    }
    *v21 = v10;
  }
  return v19;
}
