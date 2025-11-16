// Function: sub_B4E660
// Address: 0xb4e660
//
__int64 __fastcall sub_B4E660(int *a1, __int64 a2, __int64 a3)
{
  int *v4; // rbx
  __int64 v5; // rax
  __int64 **v6; // r13
  int *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 *v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r12
  __int64 **v17; // rdi
  __int64 *v18; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+18h] [rbp-B8h]
  _BYTE v20[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = a1;
  v5 = sub_BCB2D0(*(_QWORD *)a3);
  v6 = (__int64 **)v5;
  if ( *(_BYTE *)(a3 + 8) == 18 )
  {
    LODWORD(v18) = a2;
    BYTE4(v18) = 1;
    v17 = (__int64 **)sub_BCE1B0(v5, v18);
    if ( *v4 )
      return sub_ACADE0(v17);
    else
      return sub_AD6530((__int64)v17, (__int64)v18);
  }
  else
  {
    v7 = &a1[a2];
    v18 = (__int64 *)v20;
    v19 = 0x1000000000LL;
    if ( a1 == v7 )
    {
      v13 = (__int64 *)v20;
      v14 = 0;
    }
    else
    {
      do
      {
        v12 = *v4;
        if ( (_DWORD)v12 == -1 )
          v8 = sub_ACADE0(v6);
        else
          v8 = sub_AD64C0((__int64)v6, v12, 0);
        v9 = v8;
        v10 = (unsigned int)v19;
        v11 = (unsigned int)v19 + 1LL;
        if ( v11 > HIDWORD(v19) )
        {
          sub_C8D5F0(&v18, v20, v11, 8);
          v10 = (unsigned int)v19;
        }
        ++v4;
        v18[v10] = v9;
        LODWORD(v19) = v19 + 1;
      }
      while ( v7 != v4 );
      v13 = v18;
      v14 = (unsigned int)v19;
    }
    v15 = sub_AD3730(v13, v14);
    if ( v18 != (__int64 *)v20 )
      _libc_free(v18, v14);
    return v15;
  }
}
