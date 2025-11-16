// Function: sub_F7A6D0
// Address: 0xf7a6d0
//
void __fastcall sub_F7A6D0(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  char *v6; // r15
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  char *v10; // r11
  char *v11; // r10
  __int64 v12; // r14
  char *v13; // r9
  char *v14; // r10
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rdi
  char v20; // al
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // rax
  char *v24; // [rsp+10h] [rbp-70h]
  char *v25; // [rsp+18h] [rbp-68h]
  char *src; // [rsp+20h] [rbp-60h]
  char *srca; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  char *v29; // [rsp+28h] [rbp-58h]

  v28 = a1;
  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a2;
      v7 = a5;
      if ( a5 + a4 == 2 )
      {
        v13 = (char *)a1;
        v14 = a2;
LABEL_12:
        v16 = *(_QWORD *)v14;
        v17 = *(_QWORD *)v13;
        v18 = *(_QWORD *)(*(_QWORD *)v14 + 8LL);
        v19 = *(_QWORD *)(*(_QWORD *)v13 + 8LL);
        v20 = *(_BYTE *)(v19 + 8);
        if ( *(_BYTE *)(v18 + 8) == 12 )
        {
          if ( v20 != 12 )
            return;
          srca = v13;
          v29 = v14;
          v21 = sub_BCAE30(v19);
          v22 = sub_BCAE30(v18);
          v14 = v29;
          v13 = srca;
          if ( v21 >= v22 )
            return;
        }
        else if ( v20 != 12 )
        {
          return;
        }
        *(_QWORD *)v13 = v16;
        *(_QWORD *)v14 = v17;
        return;
      }
      if ( a4 <= a5 )
        goto LABEL_10;
LABEL_5:
      v8 = v5 / 2;
      v9 = sub_F79F60((__int64)v6, a3, v28 + 8 * (v5 / 2));
      v10 = (char *)(v28 + 8 * (v5 / 2));
      v11 = (char *)v9;
      v12 = (v9 - (__int64)v6) >> 3;
      while ( 1 )
      {
        v24 = v11;
        src = v10;
        v7 -= v12;
        v25 = sub_F7A510(v10, v6, v11);
        sub_F7A6D0(v28, src, v25, v8, v12);
        v5 -= v8;
        if ( !v5 )
          break;
        v13 = v25;
        v14 = v24;
        if ( !v7 )
          break;
        if ( v7 + v5 == 2 )
          goto LABEL_12;
        v28 = (__int64)v25;
        v6 = v24;
        if ( v5 > v7 )
          goto LABEL_5;
LABEL_10:
        v12 = v7 / 2;
        v15 = sub_F7A030(v28, (__int64)v6, (__int64)&v6[8 * (v7 / 2)]);
        v11 = &v6[8 * (v7 / 2)];
        v10 = (char *)v15;
        v8 = (v15 - v28) >> 3;
      }
    }
  }
}
