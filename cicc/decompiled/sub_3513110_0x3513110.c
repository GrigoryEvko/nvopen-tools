// Function: sub_3513110
// Address: 0x3513110
//
void __fastcall sub_3513110(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 *v10; // r11
  __int64 *v11; // r15
  __int64 v12; // r13
  __int64 *v13; // rax
  char *v14; // r11
  char *v15; // r9
  char *v16; // r10
  __int64 v17; // r14
  __int64 *v18; // r10
  __int64 *v19; // rax
  __int64 *v20; // r11
  __int64 v21; // r12
  unsigned int v22; // ebx
  __int64 v23; // rax
  __int64 *v25; // [rsp+8h] [rbp-58h]
  char *src; // [rsp+10h] [rbp-50h]
  __int64 *srca; // [rsp+10h] [rbp-50h]
  __int64 *v28; // [rsp+18h] [rbp-48h]
  int v29; // [rsp+18h] [rbp-48h]
  __int64 *v30; // [rsp+18h] [rbp-48h]
  __int64 *v31; // [rsp+18h] [rbp-48h]
  __int64 *v32; // [rsp+20h] [rbp-40h]

  if ( a5 )
  {
    v8 = a4;
    if ( a4 )
    {
      v9 = a5;
      if ( a5 + a4 == 2 )
      {
        v18 = a2;
        v20 = a1;
LABEL_12:
        v31 = v18;
        v21 = *v18;
        v32 = v20;
        v22 = sub_2E441D0(*(_QWORD *)(a7 + 528), *a8, *v20);
        if ( v22 < (unsigned int)sub_2E441D0(*(_QWORD *)(a7 + 528), *a8, v21) )
        {
          v23 = *v32;
          *v32 = *v31;
          *v31 = v23;
        }
      }
      else
      {
        v10 = a2;
        v11 = a1;
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v28 = v10;
        v12 = v8 / 2;
        v13 = sub_3511A00(v10, a3, &v11[v8 / 2], a7, a8);
        v14 = (char *)v28;
        v15 = (char *)&v11[v8 / 2];
        v16 = (char *)v13;
        v17 = v13 - v28;
        while ( 1 )
        {
          v25 = (__int64 *)v16;
          v29 = (int)v15;
          v9 -= v17;
          src = sub_3512D10(v15, v14, v16);
          sub_3513110((_DWORD)v11, v29, (_DWORD)src, v12, v17, v29, a7, (__int64)a8);
          v8 -= v12;
          if ( !v8 )
            break;
          v18 = v25;
          if ( !v9 )
            break;
          if ( v9 + v8 == 2 )
          {
            v20 = (__int64 *)src;
            goto LABEL_12;
          }
          v10 = v25;
          v11 = (__int64 *)src;
          if ( v9 < v8 )
            goto LABEL_5;
LABEL_10:
          v30 = v10;
          v17 = v9 / 2;
          srca = &v10[v9 / 2];
          v19 = sub_3511940(v11, (__int64)v10, srca, a7, a8);
          v16 = (char *)srca;
          v14 = (char *)v30;
          v15 = (char *)v19;
          v12 = v19 - v11;
        }
      }
    }
  }
}
