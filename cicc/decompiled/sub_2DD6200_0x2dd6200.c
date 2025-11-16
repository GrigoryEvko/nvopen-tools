// Function: sub_2DD6200
// Address: 0x2dd6200
//
void __fastcall sub_2DD6200(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  __int64 v11; // rax
  char *v12; // r10
  char *v13; // r11
  char *v14; // r11
  char *v15; // rax
  __int64 v16; // r13
  __int64 v17; // r14
  char v18; // r12
  __int64 v19; // rax
  __int64 v20; // r14
  unsigned __int64 v21; // rbx
  char v22; // r12
  __int64 v23; // rax
  char *v25; // [rsp+8h] [rbp-68h]
  char *v26; // [rsp+10h] [rbp-60h]
  char *src; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  char *v30; // [rsp+28h] [rbp-48h]

  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a1;
      v8 = a2;
      v9 = a4;
      if ( a4 + a5 == 2 )
      {
        v26 = a1;
        v14 = a2;
LABEL_12:
        v30 = v14;
        v16 = *(_QWORD *)(*(_QWORD *)v14 + 24LL);
        v17 = *(_QWORD *)v26;
        v18 = sub_AE5020(a6, v16);
        v19 = sub_9208B0(a6, v16);
        v20 = *(_QWORD *)(v17 + 24);
        v21 = ((1LL << v18) + ((unsigned __int64)(v19 + 7) >> 3) - 1) >> v18 << v18;
        v22 = sub_AE5020(a6, v20);
        if ( ((1LL << v22) + ((unsigned __int64)(sub_9208B0(a6, v20) + 7) >> 3) - 1) >> v22 << v22 > v21 )
        {
          v23 = *(_QWORD *)v26;
          *(_QWORD *)v26 = *(_QWORD *)v30;
          *(_QWORD *)v30 = v23;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v28 = v9 / 2;
        v11 = sub_2DD5A60((__int64)v8, a3, (__int64 *)&v7[8 * (v9 / 2)], a6);
        v12 = &v7[8 * (v9 / 2)];
        v13 = (char *)v11;
        v29 = (v11 - (__int64)v8) >> 3;
        while ( 1 )
        {
          v25 = v13;
          src = v12;
          v26 = sub_2DD4030(v12, v8, v13);
          sub_2DD6200(v7, src, v26, v28, v29, a6);
          v6 -= v29;
          v9 -= v28;
          if ( !v9 )
            break;
          v14 = v25;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v7 = v26;
          v8 = v25;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v29 = v6 / 2;
          v15 = (char *)sub_2DD5BA0(v7, (__int64)v8, (__int64)&v8[8 * (v6 / 2)], a6);
          v13 = &v8[8 * (v6 / 2)];
          v12 = v15;
          v28 = (v15 - v7) >> 3;
        }
      }
    }
  }
}
