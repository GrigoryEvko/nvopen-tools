// Function: sub_30B3E50
// Address: 0x30b3e50
//
__int64 __fastcall sub_30B3E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rbx
  __int64 *v7; // r15
  unsigned __int64 v8; // rdi
  void (*v9)(void); // rax
  __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 *i; // r15
  __int64 v14; // rax
  __int64 *v16; // [rsp+8h] [rbp-98h]
  __int64 *v17; // [rsp+10h] [rbp-90h] BYREF
  __int64 v18; // [rsp+18h] [rbp-88h]
  _BYTE v19[16]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v20[3]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v21; // [rsp+48h] [rbp-58h]
  _WORD *v22; // [rsp+50h] [rbp-50h]
  __int64 v23; // [rsp+58h] [rbp-48h]
  __int64 v24; // [rsp+60h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  v23 = 0x100000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v20[0] = &unk_49DD210;
  v24 = a1;
  v20[1] = 0;
  v20[2] = 0;
  v21 = 0;
  v22 = 0;
  sub_CB5980((__int64)v20, 0, 0, 0);
  v17 = (__int64 *)v19;
  v18 = 0x100000000LL;
  if ( !(unsigned __int8)sub_30B3A80(a2, a3, a4, (__int64)&v17) )
  {
LABEL_2:
    v6 = v17;
    v7 = &v17[(unsigned int)v18];
    goto LABEL_3;
  }
  v6 = &v17[(unsigned int)v18];
  v7 = v6;
  if ( v17 != v6 )
  {
    v16 = v17;
    sub_228CF00(*v17, (__int64)v20);
    v11 = *(_QWORD *)(a1 + 8);
    v12 = v16;
    if ( *(_BYTE *)(*(_QWORD *)a1 + v11 - 1) == 10 )
    {
      sub_2240CE0((__int64 *)a1, v11 - 1, 1);
      v12 = v16;
    }
    for ( i = v12 + 1; v6 != i; ++i )
    {
      if ( (unsigned __int64)(v21 - (_QWORD)v22) > 1 )
        *v22++ = 8236;
      else
        sub_CB6200((__int64)v20, (unsigned __int8 *)", ", 2u);
      sub_228CF00(*i, (__int64)v20);
      v14 = *(_QWORD *)(a1 + 8);
      if ( *(_BYTE *)(*(_QWORD *)a1 + v14 - 1) == 10 )
        sub_2240CE0((__int64 *)a1, v14 - 1, 1);
    }
    goto LABEL_2;
  }
LABEL_3:
  if ( v6 != v7 )
  {
    while ( 1 )
    {
      v8 = *--v7;
      if ( !v8 )
        goto LABEL_6;
      v9 = *(void (**)(void))(*(_QWORD *)v8 + 8LL);
      if ( (char *)v9 == (char *)sub_228A6E0 )
      {
        j_j___libc_free_0(v8);
LABEL_6:
        if ( v7 == v6 )
          goto LABEL_10;
      }
      else
      {
        v9();
        if ( v7 == v6 )
        {
LABEL_10:
          v7 = v17;
          break;
        }
      }
    }
  }
  if ( v7 != (__int64 *)v19 )
    _libc_free((unsigned __int64)v7);
  v20[0] = &unk_49DD210;
  sub_CB5840((__int64)v20);
  return a1;
}
