// Function: sub_C81DB0
// Address: 0xc81db0
//
__int64 __fastcall sub_C81DB0(const char **a1, __int64 a2)
{
  bool v2; // zf
  unsigned __int8 v3; // al
  size_t v4; // r14
  const char *v5; // r15
  unsigned int v6; // r13d
  const char *v8; // [rsp+10h] [rbp-100h] BYREF
  size_t v9; // [rsp+18h] [rbp-F8h]
  __int16 v10; // [rsp+30h] [rbp-E0h]
  const char *v11; // [rsp+40h] [rbp-D0h] BYREF
  size_t v12; // [rsp+48h] [rbp-C8h]
  __int64 v13; // [rsp+50h] [rbp-C0h]
  _BYTE v14[184]; // [rsp+58h] [rbp-B8h] BYREF

  v2 = *((_BYTE *)a1 + 33) == 1;
  v11 = v14;
  v12 = 0;
  v13 = 128;
  if ( !v2 )
    goto LABEL_8;
  v3 = *((_BYTE *)a1 + 32);
  if ( v3 == 1 )
  {
    v4 = 0;
    v5 = 0;
    goto LABEL_9;
  }
  if ( (unsigned __int8)(v3 - 3) > 3u )
  {
LABEL_8:
    sub_CA0EC0(a1, &v11);
    v4 = v12;
    v5 = v11;
    goto LABEL_9;
  }
  if ( v3 == 4 )
  {
    v5 = *(const char **)*a1;
    v4 = *((_QWORD *)*a1 + 1);
  }
  else
  {
    if ( v3 > 4u )
    {
      if ( (unsigned __int8)(v3 - 5) <= 1u )
      {
        v4 = (size_t)a1[1];
        v5 = *a1;
        goto LABEL_9;
      }
LABEL_19:
      BUG();
    }
    if ( v3 != 3 )
      goto LABEL_19;
    v5 = *a1;
    v4 = 0;
    if ( *a1 )
      v4 = strlen(*a1);
  }
LABEL_9:
  a2 = (unsigned int)a2;
  v8 = v5;
  v10 = 261;
  v9 = v4;
  v6 = sub_C81B90((__int64)&v8, a2);
  if ( (unsigned int)a2 > 1 )
  {
    a2 = (unsigned int)a2;
    v8 = v5;
    v10 = 261;
    v9 = v4;
    v6 &= sub_C81280((__int64)&v8, a2);
  }
  if ( v11 != v14 )
    _libc_free(v11, a2);
  return v6;
}
