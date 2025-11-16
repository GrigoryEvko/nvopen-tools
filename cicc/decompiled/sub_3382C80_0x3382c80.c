// Function: sub_3382C80
// Address: 0x3382c80
//
__int64 __fastcall sub_3382C80(__int64 **a1)
{
  __int64 v2; // rsi
  _QWORD *v3; // r15
  __int64 *v4; // rax
  unsigned int *v5; // rbx
  unsigned int *v6; // r14
  __int64 v7; // r12
  __int64 (*v8)(); // rax
  unsigned __int8 v9; // al
  __int64 v10; // rsi
  __int64 *v11; // rdi
  _QWORD *v12; // rsi
  char v13; // dl
  __int64 v14; // rsi
  __int64 v16; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v17; // [rsp+8h] [rbp-98h]
  _QWORD v18[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v19; // [rsp+30h] [rbp-70h]
  _QWORD v20[4]; // [rsp+40h] [rbp-60h] BYREF
  char v21; // [rsp+60h] [rbp-40h]
  char v22; // [rsp+61h] [rbp-3Fh]

  v2 = *(_QWORD *)((*a1)[108] + 40);
  v3 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 16) + 200LL))(*(_QWORD *)(v2 + 16));
  v4 = a1[1];
  v5 = (unsigned int *)v4[47];
  v6 = &v5[*((unsigned int *)v4 + 96)];
  if ( v5 == v6 )
    return 0;
  while ( 1 )
  {
    v7 = *v5;
    if ( (unsigned int)(v7 - 1) <= 0x3FFFFFFE )
    {
      v8 = *(__int64 (**)())(*v3 + 160LL);
      if ( v8 != sub_2FF51E0 )
      {
        v9 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))v8)(v3, v2, (unsigned int)v7);
        if ( v9 )
          break;
      }
    }
    if ( v6 == ++v5 )
      return 0;
  }
  v10 = v3[1];
  v11 = *a1;
  if ( *(_BYTE *)(v3[9] + *(unsigned int *)(v10 + 24 * v7)) )
  {
    v18[2] = v3[9] + *(unsigned int *)(v10 + 24 * v7);
    v18[0] = "write to reserved register '";
    v19 = 771;
  }
  else
  {
    v18[0] = "write to reserved register '";
    v19 = 259;
  }
  v12 = v18;
  v13 = 2;
  if ( HIBYTE(v19) == 1 )
  {
    v12 = (_QWORD *)v18[0];
    v13 = 3;
    v16 = v18[1];
  }
  v17 = v9;
  v20[0] = v12;
  v14 = (__int64)a1[2];
  v20[1] = v16;
  v21 = v13;
  v20[2] = "'";
  v22 = 3;
  sub_33829D0(v11, v14, (__int64)v20);
  return v17;
}
