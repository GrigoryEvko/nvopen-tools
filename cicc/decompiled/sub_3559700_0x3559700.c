// Function: sub_3559700
// Address: 0x3559700
//
void __fastcall sub_3559700(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  char *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r11
  __int64 v13; // rcx
  char *v14; // r15
  __int64 v15; // r8
  unsigned int *v16; // rax
  char *v17; // rax
  unsigned int v18; // esi
  __int64 v19; // rcx
  int v20; // esi
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  __int64 v25; // [rsp+28h] [rbp-38h]

  v21 = a3;
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
        v16 = (unsigned int *)a1;
        v14 = a2;
LABEL_13:
        v18 = *((_DWORD *)v14 + 13);
        v19 = v16[13];
        LOBYTE(a3) = v18 > (unsigned int)v19;
        if ( v18 == (_DWORD)v19 )
        {
          a3 = *((unsigned int *)v14 + 16);
          if ( !(_DWORD)a3
            || (v19 = v16[16], (_DWORD)a3 == (_DWORD)v19)
            || (LOBYTE(a3) = (unsigned int)a3 < (unsigned int)v19, !(_DWORD)v19) )
          {
            v20 = *((_DWORD *)v14 + 14);
            v19 = v16[14];
            LOBYTE(a3) = v20 < (int)v19;
            if ( v20 == (_DWORD)v19 )
              LOBYTE(a3) = *((_DWORD *)v14 + 15) > v16[15];
          }
        }
        if ( (_BYTE)a3 )
          sub_3559400((__int64)v16, (__int64)v14, a3, v19, a5, a6);
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v10 = (char *)sub_353ECD0(v8, v21, &v7[88 * (v9 / 2)]);
        v12 = (__int64)&v7[88 * (v9 / 2)];
        v13 = v9 / 2;
        v14 = v10;
        v15 = 0x2E8BA2E8BA2E8BA3LL * ((v10 - v8) >> 3);
        while ( 1 )
        {
          v24 = v15;
          v25 = v13;
          v23 = v12;
          v22 = sub_3541720(v12, (__int64)v8, (__int64)v14, v13, v15, v11);
          sub_3559700(v7, v23, v22, v25, v24);
          a5 = v24;
          v6 -= v24;
          v9 -= v25;
          if ( !v9 )
            break;
          v16 = (unsigned int *)v22;
          if ( !v6 )
            break;
          a3 = v6 + v9;
          if ( v6 + v9 == 2 )
            goto LABEL_13;
          v7 = (char *)v22;
          v8 = v14;
          if ( v6 < v9 )
            goto LABEL_5;
LABEL_10:
          v14 = &v8[88 * (v6 / 2)];
          v17 = (char *)sub_353EC20(v7, (__int64)v8, v14);
          v15 = v6 / 2;
          v12 = (__int64)v17;
          v13 = 0x2E8BA2E8BA2E8BA3LL * ((v17 - v7) >> 3);
        }
      }
    }
  }
}
