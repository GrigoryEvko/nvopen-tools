// Function: sub_22225A0
// Address: 0x22225a0
//
__int64 __fastcall sub_22225A0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9,
        _QWORD *a10)
{
  __int64 v10; // rax
  __int64 v12; // r14
  void (*v13)(void); // rax
  _BYTE *v14; // rbx
  size_t v15; // r15
  _BYTE *v16; // rdi
  size_t v17; // [rsp+28h] [rbp-58h] BYREF
  void *src; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _BYTE v20[64]; // [rsp+40h] [rbp-40h] BYREF

  v10 = *a1;
  if ( a9 )
    return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64, _QWORD))(v10 + 16))(
             a1,
             a2,
             a3,
             a4,
             a5,
             a6);
  v20[0] = 0;
  src = v20;
  n = 0;
  v12 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64, _QWORD, __int64, _DWORD *, void **))(v10 + 24))(
          a1,
          a2,
          a3,
          a4,
          a5,
          a6,
          a7,
          a8,
          &src);
  if ( !*a8 )
  {
    v13 = (void (*)(void))a10[4];
    if ( v13 )
      v13();
    v14 = src;
    v15 = n;
    v16 = a10 + 2;
    *a10 = a10 + 2;
    if ( &v14[v15] && !v14 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v17 = v15;
    if ( v15 > 0xF )
    {
      v16 = (_BYTE *)sub_22409D0(a10, &v17, 0);
      *a10 = v16;
      a10[2] = v17;
    }
    else
    {
      if ( v15 == 1 )
      {
        *((_BYTE *)a10 + 16) = *v14;
LABEL_15:
        a10[1] = v15;
        v16[v15] = 0;
        a10[4] = sub_221F8D0;
        goto LABEL_4;
      }
      if ( !v15 )
        goto LABEL_15;
    }
    memcpy(v16, v14, v15);
    v15 = v17;
    v16 = (_BYTE *)*a10;
    goto LABEL_15;
  }
LABEL_4:
  if ( src != v20 )
    j___libc_free_0((unsigned __int64)src);
  return v12;
}
