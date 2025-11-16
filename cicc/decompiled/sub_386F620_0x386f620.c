// Function: sub_386F620
// Address: 0x386f620
//
void __fastcall sub_386F620(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 *v7; // r10
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 *v10; // r15
  __int64 *v11; // rax
  char *v12; // r10
  __int64 *v13; // r14
  __int64 v14; // r13
  __int64 *v15; // r11
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v21; // [rsp+10h] [rbp-50h]
  char *v22; // [rsp+10h] [rbp-50h]
  __int64 *v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 *v25; // [rsp+20h] [rbp-40h]
  __int64 *v26; // [rsp+20h] [rbp-40h]
  __int64 v27[7]; // [rsp+28h] [rbp-38h] BYREF

  v25 = a1;
  v27[0] = a6;
  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a2;
      v8 = a4;
      if ( a5 + a4 == 2 )
      {
        v15 = a1;
        v13 = a2;
LABEL_12:
        v26 = v15;
        if ( sub_386ECF0(v27, *v13, v13[1], *v15, v15[1]) )
        {
          v17 = *v26;
          *v26 = *v13;
          v18 = v13[1];
          *v13 = v17;
          v19 = v26[1];
          v26[1] = v18;
          v13[1] = v19;
        }
      }
      else
      {
        v9 = v27[0];
        if ( v8 <= a5 )
          goto LABEL_10;
LABEL_5:
        v21 = v7;
        v24 = v8 / 2;
        v10 = &v25[2 * (v8 / 2)];
        v11 = sub_386F500(v7, a3, v10, v9);
        v12 = (char *)v21;
        v13 = v11;
        v14 = ((char *)v11 - (char *)v21) >> 4;
        while ( 1 )
        {
          v6 -= v14;
          v22 = sub_386EE20((char *)v10, v12, (char *)v13);
          sub_386F620(v25, v10, v22, v24, v14, v27[0]);
          v8 -= v24;
          if ( !v8 )
            break;
          v15 = (__int64 *)v22;
          if ( !v6 )
            break;
          if ( v6 + v8 == 2 )
            goto LABEL_12;
          v25 = (__int64 *)v22;
          v9 = v27[0];
          v7 = v13;
          if ( v8 > v6 )
            goto LABEL_5;
LABEL_10:
          v23 = v7;
          v14 = v6 / 2;
          v13 = &v7[2 * (v6 / 2)];
          v16 = sub_386F3E0(v25, (__int64)v7, v13, v9);
          v12 = (char *)v23;
          v10 = v16;
          v24 = ((char *)v16 - (char *)v25) >> 4;
        }
      }
    }
  }
}
