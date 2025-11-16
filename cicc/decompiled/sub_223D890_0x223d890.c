// Function: sub_223D890
// Address: 0x223d890
//
_QWORD *__fastcall sub_223D890(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        char a6,
        __int64 a7,
        _DWORD *a8,
        __int64 *a9)
{
  _BYTE *v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  size_t v14; // r14
  char v15; // al
  _BYTE *(__fastcall *v16)(__int64, char *, char *, void *); // rax
  int v18; // eax
  char *v19; // [rsp+8h] [rbp-68h]
  void *desta; // [rsp+10h] [rbp-60h]
  char *srca; // [rsp+18h] [rbp-58h]
  __int64 v24[7]; // [rsp+38h] [rbp-38h] BYREF

  v11 = (_BYTE *)sub_222F790((_QWORD *)(a7 + 208), a2);
  v24[0] = (__int64)&unk_4FD67D8;
  if ( a6 )
    v12 = sub_223BA00(a1, a2, a3, a4, a5, a7, a8, v24);
  else
    v12 = sub_223C8C0(a1, a2, a3, a4, a5, a7, a8, v24);
  v13 = v12;
  v14 = *(_QWORD *)(v24[0] - 24);
  srca = (char *)v24[0];
  if ( v14 )
  {
    sub_2215DA0(a9, v14, 0);
    desta = (void *)*a9;
    if ( *(int *)(*a9 - 8) >= 0 )
    {
      sub_2215730((volatile signed __int32 **)a9);
      desta = (void *)*a9;
    }
    srca = (char *)v24[0];
    v19 = (char *)(v14 + v24[0]);
    v15 = v11[56];
    if ( v15 == 1 )
    {
      if ( v19 != (char *)v24[0] )
        goto LABEL_11;
    }
    else
    {
      if ( !v15 )
        sub_2216D60((__int64)v11);
      v16 = *(_BYTE *(__fastcall **)(__int64, char *, char *, void *))(*(_QWORD *)v11 + 56LL);
      if ( v16 == sub_2216D40 )
      {
        if ( v19 != srca )
        {
LABEL_11:
          memcpy(desta, srca, v14);
          srca = (char *)v24[0];
          goto LABEL_12;
        }
      }
      else
      {
        v16((__int64)v11, srca, v19, desta);
      }
      srca = (char *)v24[0];
    }
  }
LABEL_12:
  if ( srca - 24 != (char *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v18 = _InterlockedExchangeAdd((volatile signed __int32 *)srca - 2, 0xFFFFFFFF);
    }
    else
    {
      v18 = *((_DWORD *)srca - 2);
      *((_DWORD *)srca - 2) = v18 - 1;
    }
    if ( v18 <= 0 )
      j_j___libc_free_0_1((unsigned __int64)(srca - 24));
  }
  return v13;
}
