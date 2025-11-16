// Function: sub_335DCC0
// Address: 0x335dcc0
//
__int64 __fastcall sub_335DCC0(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(void); // rdx
  __int64 result; // rax

  sub_2F8E8D0((__int64)a1, a2);
  v3 = a2[2];
  a1[73] = 0;
  a1[74] = 0;
  *a1 = &unk_4A365B8;
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 216LL);
  result = 0;
  if ( v4 != sub_2F391C0 )
    result = v4();
  a1[75] = result;
  a1[76] = 0;
  a1[77] = 0;
  a1[78] = 0;
  return result;
}
