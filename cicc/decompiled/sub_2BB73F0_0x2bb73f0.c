// Function: sub_2BB73F0
// Address: 0x2bb73f0
//
void __fastcall sub_2BB73F0(
        char *a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 (__fastcall *a7)(__int64, _QWORD, _QWORD),
        __int64 a8)
{
  __int64 v8; // r12
  __int64 v9; // rbx
  char *v10; // r10
  char *v11; // r15
  __int64 v12; // r13
  char *v13; // rax
  char *v14; // r10
  char *v15; // r11
  char *v16; // r9
  __int64 v17; // r14
  int v18; // r9d
  char *v19; // r9
  char *v20; // rax
  char *v21; // r10
  __int64 v22; // rax
  char *v24; // [rsp+8h] [rbp-58h]
  char *src; // [rsp+10h] [rbp-50h]
  char *srcb; // [rsp+10h] [rbp-50h]
  char *srca; // [rsp+10h] [rbp-50h]
  char *v28; // [rsp+18h] [rbp-48h]
  int v29; // [rsp+18h] [rbp-48h]
  char *v30; // [rsp+18h] [rbp-48h]
  char *v31; // [rsp+18h] [rbp-48h]

  if ( a4 )
  {
    v8 = a5;
    if ( a5 )
    {
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v19 = a2;
        v21 = a1;
LABEL_12:
        srca = v21;
        v31 = v19;
        if ( a7(a8, *(_QWORD *)v19, *(_QWORD *)v21) )
        {
          v22 = *(_QWORD *)srca;
          *(_QWORD *)srca = *(_QWORD *)v31;
          *(_QWORD *)v31 = v22;
        }
      }
      else
      {
        v10 = a2;
        v11 = a1;
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v28 = v10;
        v12 = v9 / 2;
        v13 = (char *)sub_2BB72F0(v10, a3, &v11[8 * (v9 / 2)], a7, a8);
        v14 = v28;
        v15 = &v11[8 * (v9 / 2)];
        v16 = v13;
        v17 = (v13 - v28) >> 3;
        while ( 1 )
        {
          v24 = v16;
          v29 = (int)v15;
          v8 -= v17;
          src = sub_2B12380(v15, v14, v16);
          sub_2BB73F0((_DWORD)v11, v29, (_DWORD)src, v12, v17, v18, (__int64)a7, a8);
          v9 -= v12;
          if ( !v9 )
            break;
          v19 = v24;
          if ( !v8 )
            break;
          if ( v8 + v9 == 2 )
          {
            v21 = src;
            goto LABEL_12;
          }
          v10 = v24;
          v11 = src;
          if ( v9 > v8 )
            goto LABEL_5;
LABEL_10:
          v30 = v10;
          v17 = v8 / 2;
          srcb = &v10[8 * (v8 / 2)];
          v20 = (char *)sub_2BB7370(v11, (__int64)v10, srcb, a7, a8);
          v16 = srcb;
          v14 = v30;
          v15 = v20;
          v12 = (v20 - v11) >> 3;
        }
      }
    }
  }
}
