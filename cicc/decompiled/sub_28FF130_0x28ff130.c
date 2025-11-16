// Function: sub_28FF130
// Address: 0x28ff130
//
__int64 __fastcall sub_28FF130(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 *v6; // [rsp+8h] [rbp-18h] BYREF

  v3 = a1;
  v6 = a2;
  if ( !sub_AA5510(a1) )
    v3 = sub_F40FB0(a1, &v6, 1, (char *)byte_3F871B3, a3, 0, 0, 0);
  return sub_F34590(v3, 0);
}
