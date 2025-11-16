// Function: sub_25FB6E0
// Address: 0x25fb6e0
//
void __fastcall sub_25FB6E0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r15
  char *v11; // r9
  char *v12; // r10
  char *v13; // r11
  __int64 v14; // r14
  int v15; // eax
  char *v16; // [rsp-58h] [rbp-58h]
  char *v17; // [rsp-50h] [rbp-50h]
  char *v18; // [rsp-48h] [rbp-48h]
  __int64 v19; // [rsp-40h] [rbp-40h]
  __int64 v20; // [rsp-30h] [rbp-30h]
  __int64 v21; // [rsp-18h] [rbp-18h]
  __int64 v22; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v22 = v7;
    v21 = v6;
    v8 = a5;
    v20 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      v15 = *(_DWORD *)a1;
      if ( *(_DWORD *)a2 < *(_DWORD *)a1 )
      {
        *(_DWORD *)a1 = *(_DWORD *)a2;
        *(_DWORD *)a2 = v15;
      }
      return;
    }
    v10 = a3;
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v12 = (char *)sub_25F68D0(a2, a3, (unsigned int *)&a1[4 * (a4 / 2)]);
      v19 = (v12 - v11) >> 2;
    }
    else
    {
      v19 = a5 / 2;
      v13 = (char *)sub_25F6880(a1, (__int64)a2, (unsigned int *)&a2[4 * (a5 / 2)]);
      v14 = (v13 - a1) >> 2;
    }
    v16 = v12;
    v17 = v13;
    v18 = sub_25FB520(v13, v11, v12);
    sub_25FB6E0(a1, v17, v18, v14, v19);
    a4 = v9 - v14;
    a3 = v10;
    v5 = v20;
    a5 = v8 - v19;
    a2 = v16;
    a1 = v18;
    v6 = v21;
    v7 = v22;
  }
}
