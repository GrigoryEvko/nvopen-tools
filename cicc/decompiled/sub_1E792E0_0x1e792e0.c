// Function: sub_1E792E0
// Address: 0x1e792e0
//
void __fastcall sub_1E792E0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 *v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 *v11; // rax
  char *v12; // r10
  char *v13; // r11
  __int64 v14; // r13
  __int64 *v15; // r11
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rax
  __int64 *v24; // r11
  __int64 v25; // rax
  __int64 *v27; // [rsp+8h] [rbp-58h]
  char *v28; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 *v31; // [rsp+20h] [rbp-40h]
  __int64 v32[7]; // [rsp+28h] [rbp-38h] BYREF

  v32[0] = a6;
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
        v28 = (char *)a1;
        v15 = a2;
LABEL_12:
        v17 = *v15;
        v18 = *(_QWORD *)(v32[0] + 280);
        v19 = *(_QWORD *)v28;
        if ( v18
          && (v31 = v15, v20 = sub_1DDC3C0(v18, *v15), v21 = *(_QWORD *)(v32[0] + 280), v22 = v20, v21)
          && (v23 = sub_1DDC3C0(v21, v19), v24 = v31, v22)
          && v23 )
        {
          if ( v22 >= v23 )
            return;
        }
        else if ( !sub_1E78020((__int64)v32, v17, v19) )
        {
          return;
        }
        v25 = *(_QWORD *)v28;
        *(_QWORD *)v28 = *v24;
        *v24 = v25;
      }
      else
      {
        v10 = v32[0];
        if ( a5 >= v9 )
          goto LABEL_10;
LABEL_5:
        v30 = v9 / 2;
        v11 = sub_1E78B90(v8, a3, &v7[v9 / 2], v10);
        v12 = (char *)&v7[v9 / 2];
        v13 = (char *)v11;
        v14 = v11 - v8;
        while ( 1 )
        {
          v27 = (__int64 *)v13;
          src = v12;
          v6 -= v14;
          v28 = sub_1E78E60(v12, (char *)v8, v13);
          sub_1E792E0(v7, src, v28, v30, v14, v32[0]);
          v9 -= v30;
          if ( !v9 )
            break;
          v15 = v27;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v7 = (__int64 *)v28;
          v10 = v32[0];
          v8 = v27;
          if ( v6 < v9 )
            goto LABEL_5;
LABEL_10:
          v14 = v6 / 2;
          v16 = sub_1E78AA0(v7, (__int64)v8, &v8[v6 / 2], v10);
          v13 = (char *)&v8[v6 / 2];
          v12 = (char *)v16;
          v30 = v16 - v7;
        }
      }
    }
  }
}
