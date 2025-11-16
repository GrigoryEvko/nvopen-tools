// Function: sub_235CD80
// Address: 0x235cd80
//
__int64 *__fastcall sub_235CD80(__int64 *a1, __int64 a2, unsigned __int64 *a3, _BYTE *a4, unsigned __int64 a5)
{
  unsigned __int64 v6; // rax
  unsigned int v8; // eax
  unsigned int v9; // r14d
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD v12[2]; // [rsp+0h] [rbp-C0h] BYREF
  const __m128i *v13; // [rsp+10h] [rbp-B0h] BYREF
  const __m128i *v14; // [rsp+18h] [rbp-A8h]
  char v15; // [rsp+28h] [rbp-98h]
  unsigned __int64 v16[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v17; // [rsp+40h] [rbp-80h] BYREF
  __int64 v18[4]; // [rsp+50h] [rbp-70h] BYREF
  char v19; // [rsp+70h] [rbp-50h]
  _QWORD v20[2]; // [rsp+78h] [rbp-48h] BYREF
  _QWORD *v21; // [rsp+88h] [rbp-38h] BYREF

  v12[0] = a4;
  v12[1] = a5;
  sub_2352D90((__int64)&v13, a4, a5);
  if ( !v15 || v13 == v14 )
  {
    v8 = sub_C63BB0();
    v18[1] = 22;
    v9 = v8;
    v11 = v10;
    v18[0] = (__int64)"invalid pipeline '{0}'";
    v18[2] = (__int64)&v21;
    v18[3] = 1;
    v19 = 1;
    v20[0] = &unk_49DB108;
    v20[1] = v12;
    v21 = v20;
    sub_23328D0((__int64)v16, (__int64)v18);
    sub_23058C0(a1, (__int64)v16, v9, v11);
    if ( (__int64 *)v16[0] != &v17 )
    {
      j_j___libc_free_0(v16[0]);
      if ( !v15 )
        return a1;
LABEL_10:
      v15 = 0;
      sub_234A6B0((unsigned __int64 *)&v13);
      return a1;
    }
  }
  else
  {
    sub_235CCD0((unsigned __int64 *)v18, a2, a3, v13, 0xCCCCCCCCCCCCCCCDLL * (((char *)v14 - (char *)v13) >> 3));
    v6 = v18[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v18[0] = 0;
      *a1 = v6 | 1;
    }
    else
    {
      v18[0] = 0;
      sub_9C66B0(v18);
      *a1 = 1;
    }
    sub_9C66B0(v18);
  }
  if ( v15 )
    goto LABEL_10;
  return a1;
}
