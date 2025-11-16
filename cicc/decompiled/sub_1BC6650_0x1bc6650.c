// Function: sub_1BC6650
// Address: 0x1bc6650
//
void __fastcall sub_1BC6650(
        char *a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 (__fastcall *a6)(_QWORD, _QWORD))
{
  __int64 v6; // r12
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  char *v11; // rax
  char *v12; // r11
  char *v13; // r10
  char *v14; // rax
  __int64 v15; // r13
  char *v16; // rdx
  char *v17; // r10
  char *v18; // rax
  __int64 v19; // rax
  char *v21; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]
  char *v24; // [rsp+20h] [rbp-40h]
  char *v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]
  char *v27; // [rsp+28h] [rbp-38h]

  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a1;
      v8 = a2;
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v17 = a2;
        v16 = a1;
LABEL_12:
        v25 = v16;
        v27 = v17;
        if ( a6(*(_QWORD *)v17, *(_QWORD *)v16) )
        {
          v19 = *(_QWORD *)v25;
          *(_QWORD *)v25 = *(_QWORD *)v27;
          *(_QWORD *)v27 = v19;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v23 = v9 / 2;
        v11 = (char *)sub_1BC6550(v8, a3, &v7[8 * (v9 / 2)], a6);
        v12 = &v7[8 * (v9 / 2)];
        v13 = v11;
        v26 = (v11 - v8) >> 3;
        while ( 1 )
        {
          v21 = v13;
          src = v12;
          v14 = sub_1BBBC60(v12, v8, v13);
          v15 = v23;
          v24 = v14;
          sub_1BC6650(v7, src, v14, v15, v26, a6);
          v6 -= v26;
          v9 -= v15;
          if ( !v9 )
            break;
          v16 = v24;
          v17 = v21;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v8 = v21;
          v7 = v24;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v26 = v6 / 2;
          v18 = (char *)sub_1BC65D0(v7, (__int64)v8, &v8[8 * (v6 / 2)], a6);
          v13 = &v8[8 * (v6 / 2)];
          v12 = v18;
          v23 = (v18 - v7) >> 3;
        }
      }
    }
  }
}
