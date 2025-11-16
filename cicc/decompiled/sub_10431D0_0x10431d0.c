// Function: sub_10431D0
// Address: 0x10431d0
//
__int64 *__fastcall sub_10431D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v7 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v8 = sub_22077B0(360);
  v9 = v8;
  if ( v8 )
    sub_1042DF0(v8, a3, v7, v6);
  *a1 = v9;
  return a1;
}
