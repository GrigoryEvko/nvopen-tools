// Function: sub_34BE2E0
// Address: 0x34be2e0
//
__int64 __fastcall sub_34BE2E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 (*v5)(); // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rdx
  __int64 result; // rax

  a1[1] = sub_B2BEC0(a3);
  *a1 = &unk_4A37808;
  v5 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
  if ( v5 == sub_23CE270 )
  {
    a1[2] = 0;
    BUG();
  }
  v6 = ((__int64 (__fastcall *)(__int64, __int64))v5)(a2, a3);
  a1[2] = v6;
  v7 = v6;
  v8 = *(__int64 (**)())(*(_QWORD *)v6 + 144LL);
  result = 0;
  if ( v8 == sub_2C8F680 )
  {
    a1[3] = 0;
  }
  else
  {
    result = ((__int64 (__fastcall *)(__int64))v8)(v7);
    a1[3] = result;
  }
  return result;
}
