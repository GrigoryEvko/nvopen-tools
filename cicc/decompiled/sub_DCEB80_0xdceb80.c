// Function: sub_DCEB80
// Address: 0xdceb80
//
__int64 __fastcall sub_DCEB80(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 *v7; // rbx
  unsigned int v8; // r15d
  unsigned __int64 v9; // r13
  _QWORD *v10; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v13; // rax
  int v14; // edx
  unsigned __int64 v15; // [rsp+8h] [rbp-E8h]
  __int64 v16; // [rsp+10h] [rbp-E0h]
  char **v17; // [rsp+80h] [rbp-70h] BYREF
  __int64 v18; // [rsp+88h] [rbp-68h]
  _BYTE v19[96]; // [rsp+90h] [rbp-60h] BYREF

  v6 = a2;
  v17 = (char **)v19;
  v16 = a4;
  v18 = 0x600000000LL;
  if ( a3 > 6 )
  {
    v11 = (unsigned __int64)v19;
    v7 = &v6[a3];
    sub_C8D5F0((__int64)&v17, v19, a3, 8u, a5, a6);
    if ( v6 == v7 )
    {
      v8 = 0;
      goto LABEL_13;
    }
  }
  else
  {
    v7 = &a2[a3];
    if ( a2 == v7 )
      return 0;
  }
  v8 = 0;
  do
  {
    v9 = *v6;
    if ( !*(_BYTE *)(a1 + 44) )
      goto LABEL_16;
    v10 = *(_QWORD **)(a1 + 24);
    v11 = *(unsigned int *)(a1 + 36);
    a3 = (unsigned __int64)&v10[v11];
    if ( v10 != (_QWORD *)a3 )
    {
      while ( v9 != *v10 )
      {
        if ( (_QWORD *)a3 == ++v10 )
          goto LABEL_23;
      }
LABEL_9:
      v8 = 1;
      goto LABEL_10;
    }
LABEL_23:
    if ( (unsigned int)v11 < *(_DWORD *)(a1 + 32) )
    {
      *(_DWORD *)(a1 + 36) = v11 + 1;
      *(_QWORD *)a3 = v9;
      ++*(_QWORD *)(a1 + 16);
    }
    else
    {
LABEL_16:
      v11 = *v6;
      sub_C8CC70(a1 + 16, *v6, a3, a4, a5, a6);
      if ( !(_BYTE)a3 )
        goto LABEL_9;
    }
    switch ( *(_WORD *)(v9 + 24) )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 0xE:
      case 0x10:
        v13 = v9;
        goto LABEL_19;
      case 9:
      case 0xA:
      case 0xB:
      case 0xC:
      case 0xD:
        v11 = v9;
        v13 = sub_DCEA90(a1, v9);
        a3 = (unsigned __int8)a3;
        if ( (_BYTE)a3 )
          goto LABEL_31;
        goto LABEL_9;
      case 0xF:
        v13 = v9;
LABEL_31:
        if ( v9 != v13 )
          v8 = 1;
LABEL_19:
        a4 = (unsigned int)v18;
        v11 = HIDWORD(v18);
        v14 = v18;
        if ( (unsigned int)v18 >= (unsigned __int64)HIDWORD(v18) )
        {
          if ( HIDWORD(v18) < (unsigned __int64)(unsigned int)v18 + 1 )
          {
            v11 = (unsigned __int64)v19;
            v15 = v13;
            sub_C8D5F0((__int64)&v17, v19, (unsigned int)v18 + 1LL, 8u, a5, a6);
            a4 = (unsigned int)v18;
            v13 = v15;
          }
          a3 = (unsigned __int64)v17;
          v17[a4] = (char *)v13;
          LODWORD(v18) = v18 + 1;
        }
        else
        {
          v11 = (unsigned __int64)v17;
          a4 = (__int64)&v17[(unsigned int)v18];
          if ( a4 )
          {
            *(_QWORD *)a4 = v13;
            v14 = v18;
          }
          a3 = (unsigned int)(v14 + 1);
          LODWORD(v18) = a3;
        }
        break;
      default:
        BUG();
    }
LABEL_10:
    ++v6;
  }
  while ( v7 != v6 );
  if ( (_BYTE)v8 )
  {
    v11 = (unsigned __int64)&v17;
    sub_D91D30(v16, (char **)&v17, a3, a4, a5, a6);
  }
LABEL_13:
  if ( v17 != (char **)v19 )
    _libc_free(v17, v11);
  return v8;
}
