// Function: sub_97B670
// Address: 0x97b670
//
__int64 __fastcall sub_97B670(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v6; // [rsp+0h] [rbp-60h] BYREF
  __int64 v7; // [rsp+8h] [rbp-58h]
  __int64 v8; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-48h]
  char v10; // [rsp+50h] [rbp-10h] BYREF

  v3 = &v8;
  v6 = 0;
  v7 = 1;
  do
  {
    *v3 = -4096;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v10 );
  if ( *a1 == 11 || (v4 = (__int64)a1, *a1 == 5) )
    v4 = sub_97B040((__int64)a1, a2, a3, (__int64)&v6);
  if ( (v7 & 1) == 0 )
    sub_C7D6A0(v8, 16LL * v9, 8);
  return v4;
}
