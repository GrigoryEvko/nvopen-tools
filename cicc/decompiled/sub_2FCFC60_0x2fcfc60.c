// Function: sub_2FCFC60
// Address: 0x2fcfc60
//
void __fastcall sub_2FCFC60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
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
  __int64 v15; // rax
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
      v15 = *(_QWORD *)a1;
      if ( *(float *)(*(_QWORD *)a2 + 116LL) > *(float *)(*(_QWORD *)a1 + 116LL) )
      {
        *(_QWORD *)a1 = *(_QWORD *)a2;
        *(_QWORD *)a2 = v15;
      }
      return;
    }
    v10 = a3;
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v12 = (char *)sub_2FCF2A0(a2, a3, a1 + 8 * (a4 / 2));
      v19 = (v12 - v11) >> 3;
    }
    else
    {
      v19 = a5 / 2;
      v13 = (char *)sub_2FCF300(a1, a2, a2 + 8 * (a5 / 2));
      v14 = (__int64)&v13[-a1] >> 3;
    }
    v16 = v12;
    v17 = v13;
    v18 = sub_2FCFAA0(v13, v11, v12);
    sub_2FCFC60(a1, v17, v18, v14, v19);
    a4 = v9 - v14;
    a3 = v10;
    v5 = v20;
    a5 = v8 - v19;
    a2 = (__int64)v16;
    a1 = (__int64)v18;
    v6 = v21;
    v7 = v22;
  }
}
