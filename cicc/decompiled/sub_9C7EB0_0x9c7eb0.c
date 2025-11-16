// Function: sub_9C7EB0
// Address: 0x9c7eb0
//
__int64 __fastcall sub_9C7EB0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4)
{
  unsigned __int64 v5; // rcx
  _BYTE *v6; // r8
  _QWORD *v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  __int64 i; // rdx
  bool v13; // zf
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  _BYTE *v17; // [rsp+0h] [rbp-80h] BYREF
  __int64 v18; // [rsp+8h] [rbp-78h]
  _BYTE v19[112]; // [rsp+10h] [rbp-70h] BYREF

  v5 = a3;
  v6 = v19;
  v17 = v19;
  v18 = 0x800000000LL;
  if ( a3 )
  {
    v9 = v19;
    if ( a3 > 8 )
    {
      sub_C8D5F0(&v17, v19, a3, 8);
      v6 = v17;
      v9 = &v17[8 * (unsigned int)v18];
    }
    v10 = 8 * a3;
    v11 = &v6[8 * a3];
    if ( v9 != v11 )
    {
      do
      {
        if ( v9 )
          *v9 = 0;
        ++v9;
      }
      while ( v11 != v9 );
      v6 = v17;
    }
    LODWORD(v18) = a3;
    if ( v10 )
    {
      for ( i = 0; i != v10; i += 8 )
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(a2 + i);
          if ( (v15 & 1) != 0 )
            break;
          *(_QWORD *)&v6[i] = v15 >> 1;
          i += 8;
          if ( v10 == i )
            goto LABEL_16;
        }
        v13 = v15 == 1;
        v14 = -(__int64)(v15 >> 1);
        if ( v13 )
          v14 = 0x8000000000000000LL;
        *(_QWORD *)&v6[i] = v14;
      }
LABEL_16:
      v6 = v17;
      v5 = (unsigned int)v18;
    }
    else
    {
      v5 = (unsigned int)a3;
    }
  }
  sub_C438C0(a1, a4, v6, v5);
  if ( v17 != v19 )
    _libc_free(v17, a4);
  return a1;
}
