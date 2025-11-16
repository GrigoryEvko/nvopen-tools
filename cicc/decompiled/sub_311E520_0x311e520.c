// Function: sub_311E520
// Address: 0x311e520
//
void __fastcall sub_311E520(unsigned __int64 **a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  char *v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rax
  char *v12; // r10
  char *v13; // r11
  __int64 v14; // r13
  char *v15; // rax
  __int64 v16; // r14
  unsigned __int64 **v17; // rdx
  unsigned __int64 **v18; // r11
  __int64 v19; // rax
  unsigned __int64 *v20; // rax
  char *v22; // [rsp+10h] [rbp-50h]
  char *srca; // [rsp+18h] [rbp-48h]
  unsigned __int64 **src; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  char *v26; // [rsp+20h] [rbp-40h]
  unsigned __int64 **v27; // [rsp+20h] [rbp-40h]
  __int64 v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v28[0] = a6;
  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = (__int64)a1;
      v8 = a2;
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v18 = (unsigned __int64 **)a2;
        v17 = a1;
LABEL_12:
        src = v17;
        v27 = v18;
        if ( (unsigned __int8)sub_311D9B0(v28, v18, v17) )
        {
          v20 = *src;
          *src = *v27;
          *v27 = v20;
        }
      }
      else
      {
        v10 = v28[0];
        if ( v9 <= a5 )
          goto LABEL_10;
LABEL_5:
        v25 = v9 / 2;
        v11 = sub_311E0B0((__int64)v8, a3, (unsigned __int64 **)(v7 + 8 * (v9 / 2)), v10);
        v12 = (char *)(v7 + 8 * (v9 / 2));
        v13 = (char *)v11;
        v14 = (v11 - (__int64)v8) >> 3;
        while ( 1 )
        {
          v22 = v13;
          srca = v12;
          v6 -= v14;
          v15 = sub_311D7F0(v12, v8, v13);
          v16 = v25;
          v26 = v15;
          sub_311E520(v7, srca, v15, v16, v14, v28[0]);
          v9 -= v16;
          if ( !v9 )
            break;
          v17 = (unsigned __int64 **)v26;
          v18 = (unsigned __int64 **)v22;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v10 = v28[0];
          v8 = v22;
          v7 = (__int64)v26;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v14 = v6 / 2;
          v19 = sub_311E130(v7, (__int64)v8, (unsigned __int64 **)&v8[8 * (v6 / 2)], v10);
          v13 = &v8[8 * (v6 / 2)];
          v12 = (char *)v19;
          v25 = (v19 - v7) >> 3;
        }
      }
    }
  }
}
