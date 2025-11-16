// Function: sub_DA4D40
// Address: 0xda4d40
//
void __fastcall sub_DA4D40(unsigned __int64 *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 *v7; // r14
  unsigned __int64 *v8; // r13
  __int64 v9; // rbx
  unsigned __int64 *v11; // rax
  char *v12; // r10
  char *v13; // r11
  unsigned __int64 *v14; // r11
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 *v18; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v19; // [rsp+10h] [rbp-60h]
  char *src; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v23; // [rsp+28h] [rbp-48h]
  __int64 v24; // [rsp+38h] [rbp-38h]

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
        v18 = a1;
        v14 = a2;
LABEL_12:
        v23 = v14;
        v24 = sub_DA4700(*(_QWORD **)a6, **(_QWORD **)(a6 + 8), *v14, *v18, *(_QWORD *)(a6 + 16), 0);
        if ( BYTE4(v24) )
        {
          if ( (int)v24 < 0 )
          {
            v16 = *v18;
            *v18 = *v23;
            *v23 = v16;
          }
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v21 = v9 / 2;
        v11 = sub_DA4CA0(v8, a3, &v7[v9 / 2], a6);
        v12 = (char *)&v7[v9 / 2];
        v13 = (char *)v11;
        v22 = v11 - v8;
        while ( 1 )
        {
          v19 = (unsigned __int64 *)v13;
          src = v12;
          v18 = (unsigned __int64 *)sub_D92500(v12, (char *)v8, v13);
          sub_DA4D40(v7, src, v18, v21, v22, a6);
          v6 -= v22;
          v9 -= v21;
          if ( !v9 )
            break;
          v14 = v19;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v7 = v18;
          v8 = v19;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v22 = v6 / 2;
          v15 = sub_DA4C00(v7, (__int64)v8, &v8[v6 / 2], a6);
          v13 = (char *)&v8[v6 / 2];
          v12 = (char *)v15;
          v21 = v15 - v7;
        }
      }
    }
  }
}
