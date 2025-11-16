// Function: sub_F7E1C0
// Address: 0xf7e1c0
//
__int64 __fastcall sub_F7E1C0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  _BYTE *i; // rbx
  unsigned __int64 v7; // rax
  _BYTE *v8; // rdi
  __int64 *v9; // r14
  __int64 *v10; // r15
  __int64 *v11; // rsi
  _BYTE *v14; // [rsp+18h] [rbp-A8h]
  _BYTE *v15; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v16; // [rsp+28h] [rbp-98h]
  _BYTE v17[32]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v18[2]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v19[96]; // [rsp+60h] [rbp-60h] BYREF

  v15 = v17;
  v16 = 0x400000000LL;
  sub_D46D90(a4, (__int64)&v15);
  v14 = &v15[8 * (unsigned int)v16];
  if ( v14 != v15 )
  {
    for ( i = v15; v14 != i; i += 8 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)i + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v7 == *(_QWORD *)i + 48LL )
        goto LABEL_23;
      if ( !v7 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
LABEL_23:
        BUG();
      if ( *(_BYTE *)(v7 - 24) == 31 && (*(_DWORD *)(v7 - 20) & 0x7FFFFFF) == 3 )
      {
        v8 = *(_BYTE **)(v7 - 120);
        if ( *v8 == 82 )
        {
          v9 = (__int64 *)*((_QWORD *)v8 - 8);
          if ( *(_BYTE *)v9 > 0x1Cu )
          {
            v10 = (__int64 *)*((_QWORD *)v8 - 4);
            if ( *(_BYTE *)v10 > 0x1Cu )
            {
              sub_B53900((__int64)v8);
              if ( a2 == sub_DD8400(*a1, (__int64)v9) )
              {
                v11 = v9;
                if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(*a1 + 40), (__int64)v9, a3) )
                {
                  LODWORD(a1) = 1;
                  goto LABEL_16;
                }
              }
              if ( a2 == sub_DD8400(*a1, (__int64)v10) )
              {
                v11 = v10;
                if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(*a1 + 40), (__int64)v10, a3) )
                {
                  LODWORD(a1) = 1;
                  goto LABEL_16;
                }
              }
            }
          }
        }
      }
    }
  }
  v11 = a2;
  v18[0] = v19;
  v18[1] = 0x600000000LL;
  LOBYTE(a1) = sub_F7DFE0((__int64)a1, (__int64)a2, a3, (__int64)v18) != 0;
  if ( (_BYTE *)v18[0] != v19 )
    _libc_free(v18[0], a2);
LABEL_16:
  if ( v15 != v17 )
    _libc_free(v15, v11);
  return (unsigned int)a1;
}
