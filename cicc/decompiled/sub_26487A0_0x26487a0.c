// Function: sub_26487A0
// Address: 0x26487a0
//
void __fastcall sub_26487A0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  char *v10; // r10
  char *v11; // r11
  char *v12; // r13
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  char *v16; // [rsp+8h] [rbp-58h]
  __int64 *v17; // [rsp+10h] [rbp-50h]
  char *v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v21[0] = a6;
  if ( a4 && a5 )
  {
    if ( a4 + a5 == 2 )
    {
      if ( sub_2648420(v21, a2, (__int64)a1) )
      {
        v14 = *a1;
        *a1 = *a2;
        *a2 = v14;
        v15 = a2[1];
        a2[1] = a1[1];
        a1[1] = v15;
      }
    }
    else
    {
      if ( a4 > a5 )
      {
        v20 = a4 / 2;
        v13 = sub_26486A0(a2, a3, (__int64)&a1[2 * (a4 / 2)], v21[0]);
        v11 = (char *)&a1[2 * (a4 / 2)];
        v10 = (char *)v13;
        v19 = ((char *)v13 - (char *)a2) >> 4;
      }
      else
      {
        v19 = a5 / 2;
        v17 = &a2[2 * (a5 / 2)];
        v9 = sub_26485A0((__int64)a1, (__int64)a2, v17, v21[0]);
        v10 = (char *)v17;
        v11 = (char *)v9;
        v20 = (v9 - (__int64)a1) >> 4;
      }
      v16 = v10;
      v18 = v11;
      v12 = sub_263E6E0(v11, (char *)a2, v10);
      sub_26487A0(a1, v18, v12, v20, v19, v21[0]);
      sub_26487A0(v12, v16, a3, a4 - v20, a5 - v19, v21[0]);
    }
  }
}
