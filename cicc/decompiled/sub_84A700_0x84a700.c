// Function: sub_84A700
// Address: 0x84a700
//
void __fastcall sub_84A700(
        __int64 a1,
        __int64 a2,
        __int64 i,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 v8; // r10
  __int64 v9; // r12
  char v10; // al
  _QWORD *v11; // r13
  char v12; // al
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+28h] [rbp-38h]

  v8 = a1;
  v9 = (__int64)a6;
  if ( a1 )
  {
    v10 = *(_BYTE *)(a1 + 80);
    if ( v10 == 16 )
    {
      v8 = **(_QWORD **)(a1 + 88);
      v10 = *(_BYTE *)(v8 + 80);
    }
    if ( v10 == 24 )
      v8 = *(_QWORD *)(v8 + 88);
    v8 = *(_QWORD *)(v8 + 88);
    for ( i = *(_QWORD *)(v8 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
LABEL_8:
    v11 = 0;
    if ( (_DWORD)a4 )
    {
      if ( !a7 )
        goto LABEL_13;
      v12 = *(_BYTE *)(a7 + 15);
      if ( a5 && v12 )
      {
        v20 = v8;
        v19 = i;
        sub_82F0D0(a7, (_DWORD *)(a5 + 68));
        sub_831410(v19, a5);
        v12 = *(_BYTE *)(a7 + 15);
        i = v19;
        v8 = v20;
      }
    }
    else
    {
      if ( !a7 )
        goto LABEL_13;
      v12 = *(_BYTE *)(a7 + 15);
    }
    v11 = (_QWORD *)a7;
    if ( v12 )
      v11 = *(_QWORD **)a7;
LABEL_13:
    v13 = 0;
    v14 = **(_QWORD **)(i + 168);
    if ( !(v14 | v9) )
      goto LABEL_32;
    while ( 1 )
    {
      v21 = v8;
      v17 = sub_84A490(v9, (__int64)v11, v14, v8);
      v8 = v21;
      v18 = v17;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = v17;
      else
        *a8 = v17;
      if ( !v9 )
        break;
      v15 = *(_QWORD *)v9;
      if ( !*(_QWORD *)v9 )
      {
        v11 = (_QWORD *)*v11;
        if ( !v14 )
          goto LABEL_32;
        goto LABEL_18;
      }
      if ( *(_BYTE *)(v15 + 8) != 3 )
      {
        v11 = (_QWORD *)*v11;
        v9 = *(_QWORD *)v9;
        if ( !v14 )
          goto LABEL_20;
        goto LABEL_18;
      }
      v15 = sub_6BBB10((_QWORD *)v9);
      v11 = (_QWORD *)*v11;
      v8 = v21;
      if ( v14 )
        goto LABEL_18;
      v9 = v15;
      v16 = v15;
LABEL_19:
      if ( !v16 )
        goto LABEL_32;
LABEL_20:
      v13 = v18;
    }
    if ( !v14 )
      goto LABEL_32;
    v15 = 0;
LABEL_18:
    v14 = *(_QWORD *)v14;
    v9 = v15;
    v16 = v15 | v14;
    goto LABEL_19;
  }
  if ( i )
    goto LABEL_8;
  if ( (_DWORD)a2 )
  {
    sub_6F41B0(a6, a2, 0, a4, a5, (__int64)a6);
    *a8 = sub_6F6D20(v9, (_DWORD *)1);
  }
  else
  {
    sub_6E5970((__int64)a6);
  }
LABEL_32:
  sub_82D8A0((_QWORD *)a7);
}
