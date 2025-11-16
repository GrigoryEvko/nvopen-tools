// Function: sub_1E37BE0
// Address: 0x1e37be0
//
void __fastcall sub_1E37BE0(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
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
  _DWORD *v15; // rax
  __int64 v16; // rax
  char *v17; // [rsp-58h] [rbp-58h]
  char *v18; // [rsp-50h] [rbp-50h]
  char *v19; // [rsp-48h] [rbp-48h]
  __int64 v20; // [rsp-40h] [rbp-40h]
  __int64 v21; // [rsp-30h] [rbp-30h]
  __int64 v22; // [rsp-18h] [rbp-18h]
  __int64 v23; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v23 = v7;
    v22 = v6;
    v8 = a5;
    v21 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      v15 = *(_DWORD **)a1;
      if ( **(_DWORD **)a2 > **(_DWORD **)a1 )
      {
        *(_QWORD *)a1 = *(_QWORD *)a2;
        *(_QWORD *)a2 = v15;
        v16 = *(_QWORD *)(a2 + 8);
        *(_QWORD *)(a2 + 8) = *((_QWORD *)a1 + 1);
        *((_QWORD *)a1 + 1) = v16;
      }
      return;
    }
    v10 = a3;
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v12 = (char *)sub_1E37560((_DWORD **)a2, a3, (unsigned int **)&a1[16 * (a4 / 2)]);
      v20 = (v12 - v11) >> 4;
    }
    else
    {
      v20 = a5 / 2;
      v13 = (char *)sub_1E375C0((_DWORD **)a1, a2, (unsigned int **)(a2 + 16 * (a5 / 2)));
      v14 = (v13 - a1) >> 4;
    }
    v17 = v12;
    v18 = v13;
    v19 = sub_1E37280(v13, v11, v12);
    sub_1E37BE0(a1, v18, v19, v14, v20);
    a4 = v9 - v14;
    a3 = v10;
    v5 = v21;
    a5 = v8 - v20;
    a2 = (__int64)v17;
    a1 = v19;
    v6 = v22;
    v7 = v23;
  }
}
