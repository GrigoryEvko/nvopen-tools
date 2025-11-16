// Function: sub_16C6310
// Address: 0x16c6310
//
__int64 __fastcall sub_16C6310(__int64 *a1, __int64 a2, __int64 a3, int *a4, __int64 a5, int a6)
{
  const char *v6; // rax
  char v10; // dl
  char v11; // dl
  _BYTE *v12; // rax
  unsigned int v13; // r13d
  _QWORD v15[2]; // [rsp+0h] [rbp-120h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-110h] BYREF
  __int16 v17; // [rsp+20h] [rbp-100h]
  _QWORD v18[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v19; // [rsp+40h] [rbp-E0h]
  _BYTE *v20; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v21; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v22[2]; // [rsp+70h] [rbp-B0h] BYREF
  _BYTE v23[160]; // [rsp+80h] [rbp-A0h] BYREF

  v6 = "-%%%%%%";
  v15[1] = a3;
  if ( a3 )
    v6 = "-%%%%%%.";
  v10 = *((_BYTE *)a1 + 16);
  v15[0] = a2;
  if ( v10 )
  {
    if ( v10 == 1 )
    {
      v16[0] = v6;
      v11 = 3;
      v17 = 259;
    }
    else
    {
      if ( *((_BYTE *)a1 + 17) == 1 )
        a1 = (__int64 *)*a1;
      else
        v10 = 2;
      v16[0] = a1;
      HIBYTE(v17) = 3;
      v16[1] = v6;
      v6 = (const char *)v16;
      LOBYTE(v17) = v10;
      v11 = 2;
    }
    v18[0] = v6;
    v18[1] = v15;
    LOBYTE(v19) = v11;
    HIBYTE(v19) = 5;
  }
  else
  {
    v17 = 256;
    v19 = 256;
  }
  v22[0] = (unsigned __int64)v23;
  v22[1] = 0x8000000000LL;
  v12 = (_BYTE *)sub_16E32E0(v18, v22);
  v21 = 257;
  if ( *v12 )
  {
    v20 = v12;
    LOBYTE(v21) = 3;
  }
  v13 = sub_16C5F40((__int64)&v20, a4, a5, 1, 0x180u, a6, 0);
  if ( (_BYTE *)v22[0] != v23 )
    _libc_free(v22[0]);
  return v13;
}
