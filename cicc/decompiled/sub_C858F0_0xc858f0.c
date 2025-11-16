// Function: sub_C858F0
// Address: 0xc858f0
//
__int64 __fastcall sub_C858F0(__int64 *a1, __int64 a2, __int64 a3, int *a4, _QWORD *a5, unsigned int a6, char a7)
{
  __int64 v7; // r10
  __int64 v8; // r11
  const char *v9; // rax
  char v13; // cl
  char v14; // cl
  _BYTE *v15; // rax
  unsigned int v16; // r13d
  const char *v18; // [rsp+0h] [rbp-150h] BYREF
  __int64 v19; // [rsp+8h] [rbp-148h]
  const char *v20; // [rsp+10h] [rbp-140h]
  __int16 v21; // [rsp+20h] [rbp-130h]
  _QWORD v22[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v23; // [rsp+50h] [rbp-100h]
  _BYTE *v24; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v25; // [rsp+80h] [rbp-D0h]
  _QWORD v26[3]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v27[168]; // [rsp+A8h] [rbp-A8h] BYREF

  v9 = "-%%%%%%";
  if ( a3 )
    v9 = "-%%%%%%.";
  v13 = *((_BYTE *)a1 + 32);
  if ( v13 )
  {
    if ( v13 == 1 )
    {
      v18 = v9;
      v14 = 3;
      v8 = v19;
      v21 = 259;
    }
    else
    {
      if ( *((_BYTE *)a1 + 33) == 1 )
      {
        v7 = a1[1];
        a1 = (__int64 *)*a1;
      }
      else
      {
        v13 = 2;
      }
      v18 = (const char *)a1;
      v19 = v7;
      HIBYTE(v21) = 3;
      v20 = v9;
      v9 = (const char *)&v18;
      LOBYTE(v21) = v13;
      v14 = 2;
    }
    v22[0] = v9;
    v22[1] = v8;
    v22[2] = a2;
    v22[3] = a3;
    LOBYTE(v23) = v14;
    HIBYTE(v23) = 5;
  }
  else
  {
    v21 = 256;
    v23 = 256;
  }
  v26[1] = 0;
  v26[0] = v27;
  v26[2] = 128;
  v15 = (_BYTE *)sub_CA12A0(v22, v26);
  v25 = 257;
  if ( *v15 )
  {
    v24 = v15;
    LOBYTE(v25) = 3;
  }
  v16 = sub_C856B0((__int64)&v24, a4, a5, 1, a6, a7, 0x1B6u);
  if ( (_BYTE *)v26[0] != v27 )
    _libc_free(v26[0], a4);
  return v16;
}
