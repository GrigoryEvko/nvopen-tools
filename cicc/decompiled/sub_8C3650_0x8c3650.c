// Function: sub_8C3650
// Address: 0x8c3650
//
void __fastcall sub_8C3650(__int64 *a1, unsigned __int8 a2, int a3)
{
  char v3; // al
  __int64 **v4; // rax
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rax
  int v10; // [rsp-2Ch] [rbp-2Ch]
  int v11; // [rsp-2Ch] [rbp-2Ch]

  if ( a1 )
  {
    v3 = *((_BYTE *)a1 - 8);
    if ( (v3 & 1) != 0 )
    {
      if ( (v3 & 2) == 0 )
      {
        sub_729FB0((__int64)a1, a2, (__int64)qword_4D03FD0);
        return;
      }
      if ( !*(a1 - 3) )
      {
        if ( a2 == 37 )
        {
          v4 = (__int64 **)a1[8];
          if ( !v4 )
          {
LABEL_16:
            v11 = a3;
            v7 = sub_7279A0(qword_4B6D500[a2]);
            *((_BYTE *)a1 - 8) |= 4u;
            v8 = (dword_4F6023C | v11) == 0;
            *(a1 - 3) = v7;
            if ( v8 )
              sub_75B140(
                (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C2C50,
                (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))sub_8C3290,
                0,
                0,
                (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C37B0,
                0,
                a1,
                a2);
            return;
          }
        }
        else
        {
          v10 = a3;
          v6 = sub_72A270((__int64)a1, a2);
          a3 = v10;
          if ( !v6 || (v4 = *(__int64 ***)(v6 + 32)) == 0 )
          {
LABEL_15:
            if ( (unsigned __int8)(a2 - 24) <= 2u )
              return;
            goto LABEL_16;
          }
        }
        v5 = *v4;
        if ( a1 != *v4 )
        {
          if ( (*(_BYTE *)(v5 - 1) & 3) == 3 )
          {
            sub_8C3650(v5, a2, 0);
            v5 = (__int64 *)*(v5 - 3);
            if ( (*(_BYTE *)(v5 - 1) & 2) != 0 )
              v5 = (__int64 *)*(v5 - 3);
          }
          *(a1 - 3) = (__int64)v5;
          return;
        }
        v9 = (__int64)v4[1];
        if ( v9 )
        {
          *(a1 - 3) = v9;
          return;
        }
        goto LABEL_15;
      }
    }
  }
}
