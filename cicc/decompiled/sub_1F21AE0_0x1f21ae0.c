// Function: sub_1F21AE0
// Address: 0x1f21ae0
//
void __fastcall sub_1F21AE0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  char *v6; // r15
  char *v7; // r10
  __int64 v8; // rbx
  __int64 v9; // r13
  char *v10; // r9
  char *v11; // r10
  char *v12; // r11
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 *v15; // r9
  char *v16; // rax
  __int64 v17; // rsi
  char *v19; // [rsp+8h] [rbp-48h]
  char *v20; // [rsp+10h] [rbp-40h]
  char *v21; // [rsp+18h] [rbp-38h]
  __int64 *v22; // [rsp+18h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a4;
      if ( a4 + a5 == 2 )
      {
        v14 = (__int64 *)a1;
        v15 = (__int64 *)a2;
LABEL_12:
        v17 = *v14;
        if ( (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) < (*(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                | (unsigned int)(*v14 >> 1)
                                                                                                & 3) )
        {
          *v14 = *v15;
          *v15 = v17;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v8 / 2;
        v10 = (char *)sub_1F20EF0(v7, a3, (__int64 *)&v6[8 * (v8 / 2)]);
        v13 = (v10 - v11) >> 3;
        while ( 1 )
        {
          v19 = v10;
          v21 = v12;
          v5 -= v13;
          v20 = sub_1F20890(v12, v11, v10);
          sub_1F21AE0(v6, v21, v20, v9, v13);
          v8 -= v9;
          if ( !v8 )
            break;
          v14 = (__int64 *)v20;
          v15 = (__int64 *)v19;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v6 = v20;
          v7 = v19;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v13 = v5 / 2;
          v22 = (__int64 *)&v7[8 * (v5 / 2)];
          v16 = (char *)sub_1F20E80(v6, (__int64)v7, v22);
          v10 = (char *)v22;
          v12 = v16;
          v9 = (v16 - v6) >> 3;
        }
      }
    }
  }
}
