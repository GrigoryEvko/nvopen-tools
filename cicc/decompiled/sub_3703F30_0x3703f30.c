// Function: sub_3703F30
// Address: 0x3703f30
//
unsigned __int64 *__fastcall sub_3703F30(unsigned __int64 *a1, _QWORD *a2, unsigned __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v7; // r15
  bool v8; // zf
  __int64 v9; // rax
  int v12; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a5[1];
  v8 = ((*(__int64 (__fastcall **)(_QWORD *))(*a2 + 48LL))(a2) & 2) == 0;
  v9 = *a2;
  if ( !v8 )
  {
    if ( a3 <= (*(__int64 (__fastcall **)(_QWORD *))(v9 + 40))(a2) )
    {
LABEL_9:
      *a5 = a2[1] + a3;
      a5[1] = a4;
      *a1 = 1;
      return a1;
    }
    goto LABEL_3;
  }
  if ( a3 > (*(__int64 (__fastcall **)(_QWORD *))(v9 + 40))(a2) )
  {
LABEL_3:
    v12 = 3;
    sub_3703E00(v13, &v12);
    goto LABEL_4;
  }
  if ( (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) >= a3 + v7 )
    goto LABEL_9;
  v12 = 1;
  sub_3703E00(v13, &v12);
LABEL_4:
  if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_9;
  *a1 = v13[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
