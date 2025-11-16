// Function: sub_2626470
// Address: 0x2626470
//
__int64 __fastcall sub_2626470(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v3; // rsi
  unsigned __int64 *v4; // r10
  unsigned __int64 v5; // r9
  unsigned __int64 *i; // rbx
  char *v7; // rsi
  size_t v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  size_t v12; // r11
  __int64 v13; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v14; // [rsp+18h] [rbp-C8h]
  size_t v15; // [rsp+20h] [rbp-C0h]
  size_t v16; // [rsp+28h] [rbp-B8h]
  __int64 v17; // [rsp+30h] [rbp-B0h]
  unsigned __int64 *v18; // [rsp+40h] [rbp-A0h]
  char v19[20]; // [rsp+50h] [rbp-90h] BYREF
  char v20; // [rsp+64h] [rbp-7Ch] BYREF
  _BYTE v21[11]; // [rsp+65h] [rbp-7Bh] BYREF
  _QWORD *v22; // [rsp+70h] [rbp-70h] BYREF
  size_t v23; // [rsp+78h] [rbp-68h]
  _QWORD v24[2]; // [rsp+80h] [rbp-60h] BYREF
  char *v25[2]; // [rsp+90h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+A0h] [rbp-40h] BYREF

  result = a2 + 8;
  v17 = *(_QWORD *)(a2 + 24);
  v13 = a2 + 8;
  if ( a2 + 8 != v17 )
  {
    do
    {
      v23 = 0;
      LOBYTE(v24[0]) = 0;
      v3 = v24;
      v22 = v24;
      v4 = *(unsigned __int64 **)(v17 + 32);
      v18 = *(unsigned __int64 **)(v17 + 40);
      if ( v18 != v4 )
      {
        v5 = *v4;
        for ( i = v4 + 1; ; ++i )
        {
          if ( v5 )
          {
            v7 = v21;
            do
            {
              *--v7 = v5 % 0xA + 48;
              v9 = v5;
              v5 /= 0xAu;
            }
            while ( v9 > 9 );
          }
          else
          {
            v20 = 48;
            v7 = &v20;
          }
          v25[0] = (char *)v26;
          sub_261A960((__int64 *)v25, v7, (__int64)v21);
          sub_2241490((unsigned __int64 *)&v22, v25[0], (size_t)v25[1]);
          if ( (_QWORD *)v25[0] != v26 )
            j_j___libc_free_0((unsigned __int64)v25[0]);
          if ( v18 == i )
            break;
          v8 = v23;
          v5 = *i;
          if ( v23 )
          {
            v10 = (unsigned __int64)v22;
            v11 = 15;
            v12 = v23 + 1;
            if ( v22 != v24 )
              v11 = v24[0];
            if ( v12 > v11 )
            {
              v14 = *i;
              v15 = v23 + 1;
              v16 = v23;
              sub_2240BB0((unsigned __int64 *)&v22, v23, 0, 0, 1u);
              v10 = (unsigned __int64)v22;
              v5 = v14;
              v12 = v15;
              v8 = v16;
            }
            *(_BYTE *)(v10 + v8) = 44;
            v23 = v12;
            *((_BYTE *)v22 + v8 + 1) = 0;
          }
        }
        v3 = v22;
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD, char *, char **))(*(_QWORD *)a1 + 120LL))(
             a1,
             v3,
             1,
             0,
             v19,
             v25) )
      {
        sub_261D790(a1, v17 + 56);
        (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
      }
      if ( v22 != v24 )
        j_j___libc_free_0((unsigned __int64)v22);
      result = sub_220EEE0(v17);
      v17 = result;
    }
    while ( v13 != result );
  }
  return result;
}
