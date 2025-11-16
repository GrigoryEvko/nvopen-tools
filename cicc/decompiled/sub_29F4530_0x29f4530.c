// Function: sub_29F4530
// Address: 0x29f4530
//
void __fastcall sub_29F4530(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  char *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r9
  __int64 v8; // rsi
  char *v9; // rdx
  __int64 v10; // r13
  char **v11; // r14
  __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // [rsp+8h] [rbp-E8h]
  __int64 v16; // [rsp+10h] [rbp-E0h]
  char *v18[3]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v19[184]; // [rsp+38h] [rbp-B8h] BYREF

  if ( a1 != a2 && a2 != a1 + 19 )
  {
    v3 = (__int64)(a1 + 19);
    while ( 1 )
    {
      v4 = a1[1];
      v5 = *(char **)v3;
      v6 = *(_QWORD *)(v3 + 8);
      v7 = *a1;
      v8 = *(_QWORD *)v3 + v4;
      if ( v4 >= v6 )
        v8 = *(_QWORD *)v3 + v6;
      v9 = (char *)*a1;
      if ( v5 != (char *)v8 )
        break;
LABEL_19:
      v4 += v7;
      if ( v9 != (char *)v4 )
        goto LABEL_11;
LABEL_20:
      v16 = v3;
      sub_29F3FE0(v3, v8, (__int64)v9, v3, v4, v7);
      v10 = v16 + 152;
LABEL_17:
      v3 = v10;
      if ( a2 == (__int64 *)v10 )
        return;
    }
    while ( *v5 >= *v9 )
    {
      if ( *v5 > *v9 )
        goto LABEL_20;
      ++v5;
      ++v9;
      if ( (char *)v8 == v5 )
        goto LABEL_19;
    }
LABEL_11:
    v18[1] = 0;
    v18[0] = v19;
    v18[2] = (char *)128;
    if ( v6 )
    {
      v15 = v3;
      sub_29F3DD0((__int64)v18, (char **)v3, (__int64)v9, v3, v4, v7);
      v3 = v15;
    }
    v10 = v3 + 152;
    v11 = (char **)v3;
    v12 = v3 - (_QWORD)a1;
    v13 = 0x86BCA1AF286BCA1BLL * ((v3 - (__int64)a1) >> 3);
    if ( v3 - (__int64)a1 > 0 )
    {
      do
      {
        v14 = (__int64)v11;
        v11 -= 19;
        sub_29F3DD0(v14, v11, v12, v3, v4, v7);
        --v13;
      }
      while ( v13 );
    }
    sub_29F3DD0((__int64)a1, v18, v12, v3, v4, v7);
    if ( v18[0] != v19 )
      _libc_free((unsigned __int64)v18[0]);
    goto LABEL_17;
  }
}
