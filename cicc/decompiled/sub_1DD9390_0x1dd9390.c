// Function: sub_1DD9390
// Address: 0x1dd9390
//
__int64 __fastcall sub_1DD9390(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 *v7; // r13
  __int64 *v8; // rdx
  char v9; // dl
  __int64 v10; // r14
  __int64 *v11; // rsi
  __int64 *v12; // rcx
  unsigned __int8 v14; // [rsp+Fh] [rbp-A1h]
  __int64 v15; // [rsp+10h] [rbp-A0h] BYREF
  __int64 *v16; // [rsp+18h] [rbp-98h]
  __int64 *v17; // [rsp+20h] [rbp-90h]
  __int64 v18; // [rsp+28h] [rbp-88h]
  int v19; // [rsp+30h] [rbp-80h]
  _BYTE v20[120]; // [rsp+38h] [rbp-78h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a1 + 8);
  if ( v5 == *(_QWORD *)(a1 + 56) + 320LL )
    v5 = 0;
  if ( a3 | a2 )
  {
    if ( a3 || !a2 )
    {
      v5 = a3;
    }
    else if ( !a4 )
    {
      v5 = 0;
    }
  }
  else
  {
    v4 = v5;
  }
  v6 = (__int64 *)v20;
  v19 = 0;
  v7 = *(__int64 **)(a1 + 88);
  v15 = 0;
  v8 = (__int64 *)v20;
  v16 = (__int64 *)v20;
  v17 = (__int64 *)v20;
  v18 = 8;
  v14 = 0;
  if ( v7 != *(__int64 **)(a1 + 96) )
  {
    while ( 1 )
    {
      v10 = *v7;
      if ( v6 != v8 )
        goto LABEL_12;
      v11 = &v6[HIDWORD(v18)];
      if ( v11 != v6 )
      {
        v12 = 0;
        while ( v10 != *v6 )
        {
          if ( *v6 == -2 )
            v12 = v6;
          if ( v11 == ++v6 )
          {
            if ( !v12 )
              goto LABEL_33;
            *v12 = v10;
            --v19;
            ++v15;
            if ( v4 != v10 )
              goto LABEL_14;
            goto LABEL_27;
          }
        }
LABEL_16:
        v14 = 1;
        v7 = (__int64 *)sub_1DD9130(a1, v7, 0);
        if ( v7 == *(__int64 **)(a1 + 96) )
          goto LABEL_28;
        goto LABEL_17;
      }
LABEL_33:
      if ( HIDWORD(v18) < (unsigned int)v18 )
      {
        ++HIDWORD(v18);
        *v11 = v10;
        ++v15;
      }
      else
      {
LABEL_12:
        sub_16CCBA0((__int64)&v15, *v7);
        if ( !v9 )
          goto LABEL_16;
      }
      if ( v4 != v10 )
      {
LABEL_14:
        if ( v5 != v10 && !*(_BYTE *)(v10 + 180) )
          goto LABEL_16;
      }
LABEL_27:
      if ( ++v7 == *(__int64 **)(a1 + 96) )
      {
LABEL_28:
        if ( v14 )
          sub_1D96570(*(unsigned int **)(a1 + 112), *(unsigned int **)(a1 + 120));
        if ( v17 != v16 )
          _libc_free((unsigned __int64)v17);
        return v14;
      }
LABEL_17:
      v8 = v17;
      v6 = v16;
    }
  }
  return v14;
}
