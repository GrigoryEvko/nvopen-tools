// Function: sub_32433D0
// Address: 0x32433d0
//
void __fastcall sub_32433D0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r13d
  int v6; // r12d
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 *v9; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-48h]
  unsigned __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-38h]

  if ( (void *)*a2 == sub_C33340() )
    sub_C3E660((__int64)&v9, (__int64)a2);
  else
    sub_C3A850((__int64)&v9, a2);
  v4 = v10;
  v5 = v10 >> 3;
  if ( (((v10 >> 3) - 4) & 0xFFFFFFFB) == 0 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 158, 0);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, v5);
    if ( *(_BYTE *)sub_31DA930(a3) )
    {
      sub_C496B0((__int64)&v11, (__int64)&v9);
      if ( v10 > 0x40 && v9 )
        j_j___libc_free_0_0((unsigned __int64)v9);
      v9 = (__int64 *)v11;
      v4 = v12;
      v10 = v12;
    }
    else
    {
      v4 = v10;
    }
    if ( v5 )
    {
      v6 = 0;
      while ( 1 )
      {
        LOBYTE(v7) = (_BYTE)v9;
        if ( v4 > 0x40 )
          v7 = *v9;
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, (unsigned __int8)v7);
        v8 = v10;
        v12 = v10;
        if ( v10 <= 0x40 )
          break;
        sub_C43780((__int64)&v11, (const void **)&v9);
        v8 = v12;
        if ( v12 <= 0x40 )
          goto LABEL_9;
        sub_C482E0((__int64)&v11, 8u);
LABEL_11:
        if ( v10 > 0x40 && v9 )
          j_j___libc_free_0_0((unsigned __int64)v9);
        ++v6;
        v9 = (__int64 *)v11;
        v4 = v12;
        v10 = v12;
        if ( v5 == v6 )
          goto LABEL_20;
      }
      v11 = (unsigned __int64)v9;
LABEL_9:
      if ( v8 == 8 )
        v11 = 0;
      else
        v11 >>= 8;
      goto LABEL_11;
    }
  }
LABEL_20:
  if ( v4 > 0x40 )
  {
    if ( v9 )
      j_j___libc_free_0_0((unsigned __int64)v9);
  }
}
