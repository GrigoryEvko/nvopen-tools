// Function: sub_27A1770
// Address: 0x27a1770
//
void __fastcall sub_27A1770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r11
  __int64 v7; // r10
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  char *v11; // r10
  __int64 v12; // r11
  char *v13; // r9
  __int64 v14; // r14
  __int64 v15; // r15
  char *v16; // rax
  int v17; // ecx
  int v18; // edx
  unsigned __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rdi
  char *v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  unsigned int *v25; // [rsp+18h] [rbp-38h]
  char *v26; // [rsp+18h] [rbp-38h]

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
        v16 = (char *)a1;
        v14 = a2;
LABEL_12:
        v17 = *(_DWORD *)v14;
        v18 = *(_DWORD *)v16;
        if ( *(_DWORD *)v14 >= *(_DWORD *)v16 )
        {
          if ( v17 != v18 )
            return;
          v19 = *((_QWORD *)v16 + 1);
          if ( *(_QWORD *)(v14 + 8) >= v19 )
            return;
        }
        else
        {
          v19 = *((_QWORD *)v16 + 1);
        }
        v20 = *((_QWORD *)v16 + 2);
        v21 = *((_QWORD *)v16 + 3);
        *(_DWORD *)v16 = v17;
        *((_QWORD *)v16 + 1) = *(_QWORD *)(v14 + 8);
        *((_QWORD *)v16 + 2) = *(_QWORD *)(v14 + 16);
        *((_QWORD *)v16 + 3) = *(_QWORD *)(v14 + 24);
        *(_DWORD *)v14 = v18;
        *(_QWORD *)(v14 + 8) = v19;
        *(_QWORD *)(v14 + 16) = v20;
        *(_QWORD *)(v14 + 24) = v21;
        return;
      }
      if ( a4 <= a5 )
        goto LABEL_10;
LABEL_5:
      v9 = v5 / 2;
      v25 = (unsigned int *)(v6 + 32 * (v5 / 2));
      v10 = sub_27A11C0(v7, a3, v25);
      v13 = (char *)v25;
      v14 = v10;
      v15 = (v10 - (__int64)v11) >> 5;
      while ( 1 )
      {
        v24 = v12;
        v26 = v13;
        v8 -= v15;
        v23 = sub_27A0CC0(v13, v11, (char *)v14);
        sub_27A1770(v24, v26, v23, v9, v15);
        v5 -= v9;
        if ( !v5 )
          break;
        v16 = v23;
        if ( !v8 )
          break;
        if ( v8 + v5 == 2 )
          goto LABEL_12;
        v6 = (__int64)v23;
        v7 = v14;
        if ( v5 > v8 )
          goto LABEL_5;
LABEL_10:
        v15 = v8 / 2;
        v14 = v7 + 32 * (v8 / 2);
        v13 = (char *)sub_27A1160(v6, v7, (unsigned int *)v14);
        v9 = (__int64)&v13[-v12] >> 5;
      }
    }
  }
}
