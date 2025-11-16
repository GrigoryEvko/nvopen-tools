// Function: sub_385C3A0
// Address: 0x385c3a0
//
void __fastcall sub_385C3A0(unsigned int *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 v6; // r12
  unsigned int *v7; // r14
  unsigned int *v8; // r13
  __int64 v9; // rbx
  unsigned int *v11; // rax
  char *v12; // r11
  char *v13; // r10
  char *v14; // rax
  __int64 v15; // r13
  _DWORD *v16; // rax
  _DWORD *v17; // r10
  unsigned int *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r8
  char *v22; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  char *v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a1;
      v8 = a2;
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v17 = a2;
        v16 = a1;
LABEL_12:
        v19 = (unsigned int)*v16;
        v20 = (unsigned int)*v17;
        if ( *(_QWORD *)(*a6 + 16 * v20) < *(_QWORD *)(*a6 + 16 * v19) )
        {
          *v16 = v20;
          *v17 = v19;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v24 = v9 / 2;
        v11 = sub_385BC10(v8, a3, &v7[v9 / 2], a6);
        v12 = (char *)&v7[v9 / 2];
        v13 = (char *)v11;
        v26 = v11 - v8;
        while ( 1 )
        {
          v22 = v13;
          src = v12;
          v14 = sub_385C1E0(v12, (char *)v8, v13);
          v15 = v24;
          v25 = v14;
          sub_385C3A0(v7, src, v14, v15, v26, a6);
          v6 -= v26;
          v9 -= v15;
          if ( !v9 )
            break;
          v16 = v25;
          v17 = v22;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v8 = (unsigned int *)v22;
          v7 = (unsigned int *)v25;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v26 = v6 / 2;
          v18 = sub_385BBB0(v7, (__int64)v8, &v8[v6 / 2], a6);
          v13 = (char *)&v8[v6 / 2];
          v12 = (char *)v18;
          v24 = v18 - v7;
        }
      }
    }
  }
}
