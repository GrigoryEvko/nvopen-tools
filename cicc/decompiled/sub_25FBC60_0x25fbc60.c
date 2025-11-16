// Function: sub_25FBC60
// Address: 0x25fbc60
//
unsigned __int64 __fastcall sub_25FBC60(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r14
  bool v9; // of
  unsigned __int8 *v10; // r11
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rbx
  unsigned __int8 **v17; // rsi
  int v18; // ecx
  unsigned __int8 **v19; // r9
  unsigned __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rbx
  int v23; // eax
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v27; // [rsp+18h] [rbp-88h]
  int v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+28h] [rbp-78h]
  int v31; // [rsp+3Ch] [rbp-64h]
  unsigned __int8 **v32; // [rsp+40h] [rbp-60h] BYREF
  __int64 v33; // [rsp+48h] [rbp-58h]
  _BYTE v34[80]; // [rsp+50h] [rbp-50h] BYREF

  v6 = 0;
  v31 = 0;
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) + 8LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 8LL) != v7 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    do
    {
      v10 = *(unsigned __int8 **)(v8 + 16);
      if ( (unsigned int)*v10 - 48 <= 5 )
      {
        v9 = __OFADD__(1, v6++);
        if ( v9 )
          v6 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v11 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
        if ( (v10[7] & 0x40) != 0 )
        {
          v12 = *((_QWORD *)v10 - 1);
          v13 = v12 + v11;
        }
        else
        {
          v12 = (__int64)&v10[-v11];
          v13 = *(_QWORD *)(v8 + 16);
        }
        v14 = v13 - v12;
        v32 = (unsigned __int8 **)v34;
        v15 = v14 >> 5;
        v33 = 0x400000000LL;
        v16 = v14 >> 5;
        if ( (unsigned __int64)v14 > 0x80 )
        {
          v29 = v14 >> 5;
          v25 = v14;
          v26 = v12;
          v27 = v10;
          sub_C8D5F0((__int64)&v32, v34, v15, 8u, v12, a6);
          v19 = v32;
          v18 = v33;
          LODWORD(v15) = v29;
          v10 = v27;
          v12 = v26;
          v14 = v25;
          v17 = &v32[(unsigned int)v33];
        }
        else
        {
          v17 = (unsigned __int8 **)v34;
          v18 = 0;
          v19 = (unsigned __int8 **)v34;
        }
        if ( v14 > 0 )
        {
          v20 = 0;
          do
          {
            v17[v20 / 8] = *(unsigned __int8 **)(v12 + 4 * v20);
            v20 += 8LL;
            --v16;
          }
          while ( v16 );
          v19 = v32;
          v18 = v33;
        }
        LODWORD(v33) = v15 + v18;
        v22 = sub_DFCEF0(a2, v10, v19, (unsigned int)(v15 + v18), 2);
        if ( v32 != (unsigned __int8 **)v34 )
        {
          v28 = v21;
          _libc_free((unsigned __int64)v32);
          v21 = v28;
        }
        v23 = 1;
        if ( v21 != 1 )
          v23 = v31;
        v9 = __OFADD__(v22, v6);
        v6 += v22;
        v31 = v23;
        if ( v9 )
        {
          v6 = 0x8000000000000000LL;
          if ( v22 > 0 )
            v6 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v8 != v7 );
  }
  return v6;
}
