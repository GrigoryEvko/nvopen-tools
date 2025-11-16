// Function: sub_2DAFBB0
// Address: 0x2dafbb0
//
_QWORD *__fastcall sub_2DAFBB0(_QWORD *a1)
{
  __int64 v1; // r11
  _QWORD *v2; // r13
  __int64 v3; // r14
  char v4; // bl
  _QWORD *v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v10; // [rsp+8h] [rbp-88h]
  _QWORD *v11; // [rsp+18h] [rbp-78h]
  unsigned int v12; // [rsp+2Ch] [rbp-64h] BYREF
  const char *v13; // [rsp+30h] [rbp-60h] BYREF
  char v14; // [rsp+50h] [rbp-40h]
  char v15; // [rsp+51h] [rbp-3Fh]

  v1 = *(a1 - 4);
  if ( *(_BYTE *)v1 == 94 )
  {
    if ( *(_DWORD *)(v1 + 80) == 1 && **(_DWORD **)(v1 + 72) == 1 && (v3 = *(_QWORD *)(v1 - 64), *(_BYTE *)v3 == 94) )
    {
      if ( (unsigned int)**(unsigned __int8 **)(v3 - 64) - 12 > 1 || *(_DWORD *)(v3 + 80) != 1 || **(_DWORD **)(v3 + 72) )
      {
        v11 = (_QWORD *)*(a1 - 4);
        v2 = 0;
        v4 = 0;
      }
      else
      {
        v2 = *(_QWORD **)(v1 - 32);
        v5 = *(_QWORD **)(v3 - 32);
        if ( *(_BYTE *)v2 != 61 )
          v2 = 0;
        v11 = (_QWORD *)*(a1 - 4);
        v4 = 1;
        if ( v5 )
        {
          sub_B43D60(a1);
LABEL_17:
          if ( !v11[2] )
            sub_B43D60(v11);
          if ( !*(_QWORD *)(v3 + 16) )
            sub_B43D60((_QWORD *)v3);
          if ( v2 && !v2[2] )
            sub_B43D60(v2);
          return v5;
        }
      }
    }
    else
    {
      v11 = (_QWORD *)*(a1 - 4);
      v2 = 0;
      v3 = 0;
      v4 = 0;
    }
  }
  else
  {
    v11 = 0;
    v2 = 0;
    v3 = 0;
    v4 = 0;
  }
  v15 = 1;
  v10 = v1;
  v13 = "exn.obj";
  v14 = 3;
  v12 = 0;
  v5 = sub_BD2C40(104, 1u);
  if ( v5 )
  {
    v6 = sub_B501B0(*(_QWORD *)(v10 + 8), &v12, 1);
    sub_B44260((__int64)v5, v6, 64, 1u, (__int64)(a1 + 3), 0);
    if ( *(v5 - 4) )
    {
      v7 = *(v5 - 3);
      *(_QWORD *)*(v5 - 2) = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = *(v5 - 2);
    }
    *(v5 - 4) = v10;
    v8 = *(_QWORD *)(v10 + 16);
    *(v5 - 3) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = v5 - 3;
    *(v5 - 2) = v10 + 16;
    *(_QWORD *)(v10 + 16) = v5 - 4;
    v5[9] = v5 + 11;
    v5[10] = 0x400000000LL;
    sub_B50030((__int64)v5, &v12, 1, (__int64)&v13);
  }
  sub_B43D60(a1);
  if ( v4 )
    goto LABEL_17;
  return v5;
}
