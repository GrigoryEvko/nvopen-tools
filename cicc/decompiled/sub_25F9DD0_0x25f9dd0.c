// Function: sub_25F9DD0
// Address: 0x25f9dd0
//
void __fastcall sub_25F9DD0(unsigned int *a1, unsigned int *a2, char *a3)
{
  __int64 v3; // rbx
  unsigned int *v4; // r12
  unsigned int *v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r8
  char *v9; // r15
  __int64 v10; // r14
  __int64 v11; // r12
  char *v12; // rdi
  signed __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r15
  char *v16; // r14
  __int64 v17; // rbx
  char *v18; // rdi
  signed __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-60h]
  signed __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  char *v25; // [rsp+20h] [rbp-40h]

  v3 = (char *)a2 - (char *)a1;
  v25 = &a3[(char *)a2 - (char *)a1];
  v24 = 0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)a1) >> 3);
  if ( (char *)a2 - (char *)a1 <= 912 )
  {
    sub_25F98B0(a1, a2);
  }
  else
  {
    v4 = a1;
    do
    {
      v5 = v4;
      v4 += 266;
      sub_25F98B0(v5, v4);
    }
    while ( (char *)a2 - (char *)v4 > 912 );
    sub_25F98B0(v4, a2);
    if ( v3 > 1064 )
    {
      v6 = 7;
      while ( 1 )
      {
        v7 = 2 * v6;
        if ( v24 < 2 * v6 )
        {
          v8 = (__int64)a3;
          v13 = v24;
          v9 = (char *)a1;
        }
        else
        {
          v8 = (__int64)a3;
          v9 = (char *)a1;
          v20 = v6;
          v10 = 304 * v6;
          v11 = 152 * v6;
          do
          {
            v12 = v9;
            v9 += v10;
            v8 = sub_25F76A0(v12, &v9[v11 - v10], &v9[v11 - v10], v9, v8);
            v13 = 0x86BCA1AF286BCA1BLL * (((char *)a2 - v9) >> 3);
          }
          while ( v7 <= v13 );
          v6 = v20;
        }
        if ( v6 <= v13 )
          v13 = v6;
        v6 *= 4;
        sub_25F76A0(v9, &v9[152 * v13], &v9[152 * v13], (char *)a2, v8);
        v14 = (__int64)a1;
        if ( v24 < v6 )
          break;
        v21 = v7;
        v15 = 152 * v6;
        v16 = a3;
        v17 = 152 * v7;
        do
        {
          v18 = v16;
          v16 += v15;
          v14 = sub_25F7560(v18, &v16[v17 - v15], &v16[v17 - v15], v16, v14);
          v19 = 0x86BCA1AF286BCA1BLL * ((v25 - v16) >> 3);
        }
        while ( v6 <= v19 );
        if ( v19 > v21 )
          v19 = v21;
        sub_25F7560(v16, &v16[152 * v19], &v16[152 * v19], v25, v14);
        if ( v24 <= v6 )
          return;
      }
      if ( v24 <= v7 )
        v7 = v24;
      sub_25F7560(a3, &a3[152 * v7], &a3[152 * v7], v25, (__int64)a1);
    }
  }
}
