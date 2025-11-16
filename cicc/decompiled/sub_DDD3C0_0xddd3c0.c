// Function: sub_DDD3C0
// Address: 0xddd3c0
//
__int64 __fastcall sub_DDD3C0(__int64 *a1, unsigned __int8 *a2)
{
  char v3; // dl
  int v5; // eax
  int v6; // r14d
  int v7; // eax
  bool v8; // r13
  __int64 *v9; // r15
  __int64 *v10; // r8
  int v11; // esi
  char v12; // al
  __int64 v13; // r9
  __int64 v14; // r9
  char v15; // al
  int v16; // esi
  int v17; // esi
  char v18; // al
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v3 = (a2[1] >> 1) & 2;
  if ( (a2[1] & 2) == 0 )
  {
    v5 = *a2;
    v6 = 4 * (v3 != 0);
    if ( (unsigned __int8)v5 <= 0x1Cu )
    {
LABEL_22:
      v7 = *((unsigned __int16 *)a2 + 1);
LABEL_7:
      v8 = v7 != 17 && (v7 & 0xFFFFFFFD) != 13;
      if ( v8 )
        goto LABEL_3;
      v9 = sub_DD8400((__int64)a1, *((_QWORD *)a2 - 8));
      v10 = sub_DD8400((__int64)a1, *((_QWORD *)a2 - 4));
      if ( (_BYTE)qword_4F88968 )
      {
        v11 = *a2;
        v12 = a2[1] >> 1;
        if ( (unsigned __int8)v11 > 0x1Cu )
        {
          v13 = (__int64)a2;
          if ( (a2[1] & 2) != 0 )
          {
            if ( (v12 & 2) != 0 )
              goto LABEL_3;
            v14 = (__int64)a2;
            goto LABEL_17;
          }
          goto LABEL_25;
        }
        if ( (a2[1] & 2) != 0 )
        {
          if ( (v12 & 2) != 0 )
            goto LABEL_3;
          v14 = 0;
          goto LABEL_31;
        }
      }
      else
      {
        v15 = a2[1] >> 1;
        if ( (a2[1] & 2) != 0 )
        {
          v14 = 0;
          goto LABEL_15;
        }
        v11 = *a2;
        if ( (unsigned __int8)v11 > 0x1Cu )
        {
          v13 = 0;
LABEL_25:
          v17 = v11 - 29;
LABEL_26:
          v19 = v13;
          v20 = v10;
          v18 = sub_DDCBC0(a1, v17, 0, (__int64)v9, (__int64)v10, v13);
          v10 = v20;
          v14 = v19;
          if ( v18 )
          {
            v6 |= 2u;
            if ( (a2[1] & 4) != 0 )
              goto LABEL_20;
            v8 = v18;
LABEL_16:
            v11 = *a2;
            if ( (unsigned __int8)v11 > 0x1Cu )
            {
LABEL_17:
              v16 = v11 - 29;
              goto LABEL_18;
            }
LABEL_31:
            v16 = *((unsigned __int16 *)a2 + 1);
LABEL_18:
            if ( (unsigned __int8)sub_DDCBC0(a1, v16, 1, (__int64)v9, (__int64)v10, v14) )
            {
              v6 |= 4u;
            }
            else if ( !v8 )
            {
              goto LABEL_3;
            }
LABEL_20:
            LODWORD(v21) = v6;
            BYTE4(v21) = 1;
            return v21;
          }
          v15 = a2[1] >> 1;
LABEL_15:
          if ( (v15 & 2) != 0 )
            goto LABEL_3;
          goto LABEL_16;
        }
      }
      v17 = *((unsigned __int16 *)a2 + 1);
      v13 = 0;
      goto LABEL_26;
    }
LABEL_6:
    v7 = v5 - 29;
    goto LABEL_7;
  }
  if ( !v3 )
  {
    v5 = *a2;
    v6 = 2;
    if ( (unsigned __int8)v5 <= 0x1Cu )
      goto LABEL_22;
    goto LABEL_6;
  }
LABEL_3:
  BYTE4(v21) = 0;
  return v21;
}
