// Function: sub_161DD90
// Address: 0x161dd90
//
void __fastcall sub_161DD90(char *a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  char *v6; // r11
  _DWORD *v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  char *v10; // r9
  char *v11; // r10
  __int64 v12; // r11
  _DWORD *v13; // r15
  __int64 v14; // r14
  char *v15; // rax
  int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rdx
  char *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  char *v22; // [rsp+18h] [rbp-38h]

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
        v15 = a1;
        v13 = a2;
LABEL_12:
        v16 = *(_DWORD *)v15;
        if ( *v13 < *(_DWORD *)v15 )
        {
          *(_DWORD *)v15 = *v13;
          v17 = *((_QWORD *)v13 + 1);
          *v13 = v16;
          v18 = *((_QWORD *)v15 + 1);
          *((_QWORD *)v15 + 1) = v17;
          *((_QWORD *)v13 + 1) = v18;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_161DB20(v7, a3, &v6[16 * (v5 / 2)]);
        v14 = ((char *)v13 - v10) >> 4;
        while ( 1 )
        {
          v21 = v12;
          v22 = v11;
          v8 -= v14;
          v20 = sub_161CF80(v11, v10, (char *)v13);
          sub_161DD90(v21, v22, v20, v9, v14);
          v5 -= v9;
          if ( !v5 )
            break;
          v15 = v20;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = v20;
          v7 = v13;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v8 / 2;
          v13 = &v7[4 * (v8 / 2)];
          v11 = (char *)sub_161DB70(v6, (__int64)v7, v13);
          v9 = (__int64)&v11[-v12] >> 4;
        }
      }
    }
  }
}
