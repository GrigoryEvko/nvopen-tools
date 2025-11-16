// Function: sub_1255270
// Address: 0x1255270
//
unsigned __int64 *__fastcall sub_1255270(
        unsigned __int64 *a1,
        _QWORD *a2,
        unsigned __int64 a3,
        const void *a4,
        size_t a5)
{
  bool v8; // zf
  __int64 v9; // rax
  __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !a5 )
    goto LABEL_8;
  v8 = ((*(__int64 (__fastcall **)(_QWORD *))(*a2 + 48LL))(a2) & 2) == 0;
  v9 = *a2;
  if ( !v8 )
  {
    if ( a3 <= (*(__int64 (__fastcall **)(_QWORD *))(v9 + 40))(a2) )
    {
LABEL_7:
      memcpy((void *)(a3 + a2[1]), a4, a5);
LABEL_8:
      *a1 = 1;
      return a1;
    }
    goto LABEL_4;
  }
  if ( a3 > (*(__int64 (__fastcall **)(_QWORD *))(v9 + 40))(a2) )
  {
LABEL_4:
    sub_1254FA0(v11, 3);
    goto LABEL_5;
  }
  if ( (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) >= a5 + a3 )
    goto LABEL_7;
  sub_1254FA0(v11, 1);
LABEL_5:
  if ( (v11[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_7;
  *a1 = v11[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
