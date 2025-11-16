// Function: sub_19FFF00
// Address: 0x19fff00
//
void __fastcall sub_19FFF00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r10
  __int64 v8; // r12
  __int64 v9; // r13
  char *v10; // r9
  char *v11; // r10
  char *v12; // r11
  __int64 v13; // r14
  char *v14; // rax
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // rdx
  char *v19; // [rsp+8h] [rbp-48h]
  char *v20; // [rsp+10h] [rbp-40h]
  char *v21; // [rsp+18h] [rbp-38h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = a2;
        v14 = (char *)a1;
LABEL_12:
        v17 = *(_QWORD *)v14;
        if ( *(_DWORD *)(*(_QWORD *)v15 + 32LL) < *(_DWORD *)(*(_QWORD *)v14 + 32LL) )
        {
          *(_QWORD *)v14 = *(_QWORD *)v15;
          *(_QWORD *)v15 = v17;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v10 = (char *)sub_19FED00(v7, a3, v6 + 8 * (v5 / 2));
        v13 = (v10 - v11) >> 3;
        while ( 1 )
        {
          v19 = v10;
          v21 = v12;
          v8 -= v13;
          v20 = sub_19FF890(v12, v11, v10);
          sub_19FFF00(v6, v21, v20, v9, v13);
          v5 -= v9;
          if ( !v5 )
            break;
          v14 = v20;
          v15 = (__int64)v19;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v7 = (__int64)v19;
          v6 = (__int64)v20;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v13 = v8 / 2;
          v22 = v7 + 8 * (v8 / 2);
          v16 = sub_19FED50(v6, v7, v22);
          v10 = (char *)v22;
          v12 = (char *)v16;
          v9 = (v16 - v6) >> 3;
        }
      }
    }
  }
}
