// Function: sub_23FCCD0
// Address: 0x23fccd0
//
void __fastcall sub_23FCCD0(__int64 ***a1, __int64 ***a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 ***v8; // r8
  __int64 ***v9; // r15
  __int64 v10; // r13
  __int64 ***v11; // rax
  char *v12; // r8
  char *v13; // r11
  char *v14; // r10
  __int64 v15; // r14
  __int64 ***v16; // r10
  __int64 ***v17; // rax
  __int64 ***v18; // rdx
  __int64 **v19; // r12
  unsigned int v20; // ebx
  __int64 **v21; // rax
  char *v23; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 ***srca; // [rsp+18h] [rbp-48h]
  __int64 ***v27; // [rsp+20h] [rbp-40h]
  char *v28; // [rsp+28h] [rbp-38h]
  char *v29; // [rsp+28h] [rbp-38h]
  char *v30; // [rsp+28h] [rbp-38h]
  __int64 ***v31; // [rsp+28h] [rbp-38h]

  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a4;
      if ( a5 + a4 == 2 )
      {
        v16 = a2;
        v18 = a1;
LABEL_12:
        v19 = *v18;
        v27 = v18;
        v31 = v16;
        v20 = sub_22DADF0(***v16);
        if ( v20 < (unsigned int)sub_22DADF0(**v19) )
        {
          v21 = *v27;
          *v27 = *v31;
          *v31 = v21;
        }
      }
      else
      {
        v8 = a2;
        v9 = a1;
        if ( a4 <= v6 )
          goto LABEL_10;
LABEL_5:
        v28 = (char *)v8;
        v10 = v7 / 2;
        v11 = sub_23FB300(v8, a3, &v9[v7 / 2]);
        v12 = v28;
        v13 = (char *)&v9[v7 / 2];
        v14 = (char *)v11;
        v15 = ((char *)v11 - v28) >> 3;
        while ( 1 )
        {
          v23 = v14;
          v29 = v13;
          v6 -= v15;
          src = sub_23FB780(v13, v12, v14);
          sub_23FCCD0(v9, v29, src, v10, v15, a6);
          v7 -= v10;
          if ( !v7 )
            break;
          v16 = (__int64 ***)v23;
          if ( !v6 )
            break;
          if ( v6 + v7 == 2 )
          {
            v18 = (__int64 ***)src;
            goto LABEL_12;
          }
          v8 = (__int64 ***)v23;
          v9 = (__int64 ***)src;
          if ( v7 > v6 )
            goto LABEL_5;
LABEL_10:
          v30 = (char *)v8;
          v15 = v6 / 2;
          srca = &v8[v6 / 2];
          v17 = sub_23FB260(v9, (__int64)v8, srca);
          v14 = (char *)srca;
          v12 = v30;
          v13 = (char *)v17;
          v10 = v17 - v9;
        }
      }
    }
  }
}
