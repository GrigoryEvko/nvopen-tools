// Function: sub_2FC01B0
// Address: 0x2fc01b0
//
void __fastcall sub_2FC01B0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // r12
  char *v11; // rax
  char *v12; // r11
  char *v13; // r10
  char *v14; // rax
  __int64 v15; // r13
  char *v16; // rax
  char *v17; // r10
  char *v18; // rax
  int v19; // edx
  int v20; // ecx
  char *v22; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  char *v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  if ( a5 )
  {
    v6 = a4;
    if ( a4 )
    {
      v7 = a1;
      v8 = a2;
      v9 = a5;
      if ( a5 + a4 == 2 )
      {
        v17 = a2;
        v16 = a1;
LABEL_12:
        v19 = *(_DWORD *)v17;
        if ( *(_DWORD *)v17 != -1 )
        {
          v20 = *(_DWORD *)v16;
          if ( *(_DWORD *)v16 == -1
            || *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a6 + 8LL)
                         + 40LL * (unsigned int)(v19 + *(_DWORD *)(*(_QWORD *)a6 + 32LL))
                         + 8) > *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a6 + 8LL)
                                          + 40LL * (unsigned int)(v20 + *(_DWORD *)(*(_QWORD *)a6 + 32LL))
                                          + 8) )
          {
            *(_DWORD *)v16 = v19;
            *(_DWORD *)v17 = v20;
          }
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v24 = v6 / 2;
        v11 = (char *)sub_2FBF0F0(v8, a3, (int *)&v7[4 * (v6 / 2)], a6);
        v12 = &v7[4 * (v6 / 2)];
        v13 = v11;
        v26 = (v11 - v8) >> 2;
        while ( 1 )
        {
          v22 = v13;
          src = v12;
          v14 = sub_2FBFB00(v12, v8, v13);
          v15 = v24;
          v25 = v14;
          sub_2FC01B0(v7, src, v14, v15, v26, a6);
          v9 -= v26;
          v6 -= v15;
          if ( !v6 )
            break;
          v16 = v25;
          v17 = v22;
          if ( !v9 )
            break;
          if ( v9 + v6 == 2 )
            goto LABEL_12;
          v8 = v22;
          v7 = v25;
          if ( v6 > v9 )
            goto LABEL_5;
LABEL_10:
          v26 = v9 / 2;
          v18 = (char *)sub_2FBF060(v7, (__int64)v8, (int *)&v8[4 * (v9 / 2)], a6);
          v13 = &v8[4 * (v9 / 2)];
          v12 = v18;
          v24 = (v18 - v7) >> 2;
        }
      }
    }
  }
}
