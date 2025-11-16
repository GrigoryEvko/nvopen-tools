// Function: sub_2B13E30
// Address: 0x2b13e30
//
void __fastcall sub_2B13E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  char *v10; // r9
  char *v11; // r10
  __int64 v12; // r11
  __int64 v13; // r15
  __int64 v14; // r14
  char *v15; // rax
  __int64 v16; // rdx
  int v17; // ecx
  char *v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  char *v21; // [rsp+18h] [rbp-38h]

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
        v15 = (char *)a1;
        v13 = a2;
LABEL_12:
        if ( *(_DWORD *)(v13 + 8) < *((_DWORD *)v15 + 2) )
        {
          v16 = *(_QWORD *)v15;
          *(_QWORD *)v15 = *(_QWORD *)v13;
          v17 = *(_DWORD *)(v13 + 8);
          *(_QWORD *)v13 = v16;
          LODWORD(v16) = *((_DWORD *)v15 + 2);
          *((_DWORD *)v15 + 2) = v17;
          *(_DWORD *)(v13 + 8) = v16;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_2B0F0D0(v7, a3, v6 + 16 * (v5 / 2));
        v14 = (v13 - (__int64)v10) >> 4;
        while ( 1 )
        {
          v20 = v12;
          v21 = v11;
          v8 -= v14;
          v19 = sub_2B09FA0(v11, v10, (char *)v13);
          sub_2B13E30(v20, v21, v19, v9, v14);
          v5 -= v9;
          if ( !v5 )
            break;
          v15 = v19;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = (__int64)v19;
          v7 = v13;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v8 / 2;
          v13 = v7 + 16 * (v8 / 2);
          v11 = (char *)sub_2B0F080(v6, v7, v13);
          v9 = (__int64)&v11[-v12] >> 4;
        }
      }
    }
  }
}
