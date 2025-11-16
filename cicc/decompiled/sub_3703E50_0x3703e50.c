// Function: sub_3703E50
// Address: 0x3703e50
//
unsigned __int64 *__fastcall sub_3703E50(unsigned __int64 *a1, _QWORD *a2, unsigned __int64 a3, _QWORD *a4)
{
  bool v6; // zf
  __int64 v7; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  int v11; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = ((*(__int64 (__fastcall **)(_QWORD *))(*a2 + 48LL))(a2) & 2) == 0;
  v7 = *a2;
  if ( v6 )
  {
    if ( a3 <= (*(__int64 (__fastcall **)(_QWORD *))(v7 + 40))(a2) )
    {
      if ( (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) >= a3 + 1 )
        goto LABEL_8;
      v11 = 1;
      sub_3703E00(v12, &v11);
      goto LABEL_4;
    }
  }
  else if ( a3 <= (*(__int64 (__fastcall **)(_QWORD *))(v7 + 40))(a2) )
  {
    goto LABEL_8;
  }
  v11 = 3;
  sub_3703E00(v12, &v11);
LABEL_4:
  if ( (v12[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v12[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
LABEL_8:
  v9 = a2[1];
  v10 = a2[2] - v9 - a3;
  *a4 = a3 + v9;
  a4[1] = v10;
  *a1 = 1;
  return a1;
}
