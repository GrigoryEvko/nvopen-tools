// Function: sub_26F15D0
// Address: 0x26f15d0
//
void __fastcall sub_26F15D0(__int64 a1, __int64 a2, char a3)
{
  __int64 *v3; // rbx
  __int64 *i; // r13
  const char *v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 *v14; // [rsp+20h] [rbp-90h] BYREF
  __int64 v15; // [rsp+28h] [rbp-88h]
  _BYTE v16[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v17; // [rsp+50h] [rbp-60h] BYREF
  __int64 v18; // [rsp+58h] [rbp-58h]
  _BYTE v19[80]; // [rsp+60h] [rbp-50h] BYREF

  v14 = (__int64 *)v16;
  v15 = 0x400000000LL;
  v17 = v19;
  v18 = 0x400000000LL;
  sub_BAA9B0(a1, (__int64)&v14, a3);
  v3 = &v14[(unsigned int)v15];
  for ( i = v14; v3 != i; LODWORD(v18) = v18 + 1 )
  {
    while ( 1 )
    {
      v5 = sub_BD5D20(*i);
      v7 = sub_BA8B30(a2, (__int64)v5, v6);
      v8 = v7;
      if ( v7 )
      {
        if ( !sub_B2FC80(v7) )
          break;
      }
      if ( v3 == ++i )
        goto LABEL_9;
    }
    v11 = (unsigned int)v18;
    v12 = (unsigned int)v18 + 1LL;
    if ( v12 > HIDWORD(v18) )
    {
      sub_C8D5F0((__int64)&v17, v19, v12, 8u, v9, v10);
      v11 = (unsigned int)v18;
    }
    ++i;
    *(_QWORD *)&v17[8 * v11] = v8;
  }
LABEL_9:
  if ( a3 )
    sub_2A41DC0(a2, v17, (unsigned int)v18);
  else
    sub_2A413E0(a2, v17, (unsigned int)v18);
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
  if ( v14 != (__int64 *)v16 )
    _libc_free((unsigned __int64)v14);
}
