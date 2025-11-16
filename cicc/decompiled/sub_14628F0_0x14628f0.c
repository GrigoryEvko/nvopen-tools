// Function: sub_14628F0
// Address: 0x14628f0
//
void __fastcall sub_14628F0(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 **v12; // r8
  __int64 *v13; // r15
  __int64 v14; // r13
  __int64 **v15; // rax
  char *v16; // r8
  char *v17; // r11
  char *v18; // r10
  __int64 v19; // r14
  int v20; // r9d
  __int64 *v21; // rax
  __int64 **v22; // r10
  __int64 *v23; // rax
  __int64 *v24; // rbx
  __int64 *v25; // rax
  char *v27; // [rsp+8h] [rbp-A8h]
  char *src; // [rsp+10h] [rbp-A0h]
  __int64 **srca; // [rsp+10h] [rbp-A0h]
  char *v30; // [rsp+18h] [rbp-98h]
  int v31; // [rsp+18h] [rbp-98h]
  char *v32; // [rsp+18h] [rbp-98h]
  __int64 **v33; // [rsp+30h] [rbp-80h]

  if ( a5 )
  {
    v10 = a4;
    if ( a4 )
    {
      v11 = a5;
      if ( a5 + a4 == 2 )
      {
        v22 = a2;
        v21 = a1;
LABEL_12:
        v24 = v21;
        v33 = v22;
        if ( (int)sub_1462150(a7, a8, *a9, *v22, *v21, a10, 0) < 0 )
        {
          v25 = (__int64 *)*v24;
          *v24 = (__int64)*v33;
          *v33 = v25;
        }
      }
      else
      {
        v12 = a2;
        v13 = a1;
        if ( v11 >= a4 )
          goto LABEL_10;
LABEL_5:
        v30 = (char *)v12;
        v14 = v10 / 2;
        v15 = sub_1462770(v12, a3, &v13[v10 / 2], a4, (__int64)v12, a6, a7, a8, a9, a10);
        v16 = v30;
        v17 = (char *)&v13[v10 / 2];
        v18 = (char *)v15;
        v19 = ((char *)v15 - v30) >> 3;
        while ( 1 )
        {
          v27 = v18;
          v31 = (int)v17;
          v11 -= v19;
          src = sub_14543A0(v17, v16, v18);
          sub_14628F0((_DWORD)v13, v31, (_DWORD)src, v14, v19, v20, (__int64)a7, (__int64)a8, (__int64)a9, a10);
          v10 -= v14;
          if ( !v10 )
            break;
          v21 = (__int64 *)src;
          v22 = (__int64 **)v27;
          if ( !v11 )
            break;
          if ( v11 + v10 == 2 )
            goto LABEL_12;
          v12 = (__int64 **)v27;
          v13 = (__int64 *)src;
          if ( v11 < v10 )
            goto LABEL_5;
LABEL_10:
          v32 = (char *)v12;
          v19 = v11 / 2;
          srca = &v12[v11 / 2];
          v23 = sub_1462830(v13, (__int64)v12, srca, a4, (__int64)v12, a6, a7, a8, a9, a10);
          v18 = (char *)srca;
          v16 = v32;
          v17 = (char *)v23;
          v14 = v23 - v13;
        }
      }
    }
  }
}
