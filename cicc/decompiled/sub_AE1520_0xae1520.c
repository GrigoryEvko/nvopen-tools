// Function: sub_AE1520
// Address: 0xae1520
//
__int64 __fastcall sub_AE1520(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        int a9,
        int a10,
        __int16 a11)
{
  char v11; // al
  __int64 *v12; // rdx
  char v13; // dl
  const char *v14; // rax
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // r13
  _QWORD v19[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v21[4]; // [rsp+20h] [rbp-80h] BYREF
  __int16 v22; // [rsp+40h] [rbp-60h]
  _QWORD v23[4]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v24; // [rsp+70h] [rbp-30h]

  v11 = a11;
  if ( (_BYTE)a11 )
  {
    if ( (_BYTE)a11 == 1 )
    {
      v14 = "malformed specification, must be of the form \"";
      a4 = v21[1];
      v22 = 259;
      v13 = 3;
      v21[0] = "malformed specification, must be of the form \"";
    }
    else
    {
      if ( HIBYTE(a11) == 1 )
      {
        a2 = a8;
        v12 = a7;
      }
      else
      {
        v12 = (__int64 *)&a7;
        v11 = 2;
      }
      v21[3] = a2;
      v21[0] = "malformed specification, must be of the form \"";
      LOBYTE(v22) = 3;
      v21[2] = v12;
      v13 = 2;
      HIBYTE(v22) = v11;
      v14 = (const char *)v21;
    }
    v23[0] = v14;
    v23[1] = a4;
    v23[2] = "\"";
    LOBYTE(v24) = v13;
    HIBYTE(v24) = 3;
  }
  else
  {
    v22 = 256;
    v24 = 256;
  }
  v15 = sub_C63BB0();
  v17 = v16;
  sub_CA0F50(v19, v23);
  sub_C63F00(a1, v19, v15, v17);
  if ( (__int64 *)v19[0] != &v20 )
    j_j___libc_free_0(v19[0], v20 + 1);
  return a1;
}
