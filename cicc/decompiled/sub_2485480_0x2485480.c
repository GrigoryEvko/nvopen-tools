// Function: sub_2485480
// Address: 0x2485480
//
__int64 __fastcall sub_2485480(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  _BYTE *v4; // rdi
  __int64 v5; // rax
  __int64 v7; // [rsp+8h] [rbp-7B8h] BYREF
  _BYTE v8[1920]; // [rsp+10h] [rbp-7B0h] BYREF
  _BYTE *v9; // [rsp+790h] [rbp-30h]

  v2 = a1;
  v3 = &a1[4 * a2];
  sub_CC1970((__int64)v8);
  v9 = v8;
  v4 = v8;
  for ( v8[1912] = 1; v3 != v2; v4 = v9 )
  {
    v5 = *v2;
    v2 += 4;
    v7 = v5;
    sub_CC19D0((__int64)v4, &v7, 8u);
    LODWORD(v7) = *((_DWORD *)v2 - 4);
    sub_CC19D0((__int64)v9, &v7, 4u);
    LODWORD(v7) = *((_DWORD *)v2 - 3);
    sub_CC19D0((__int64)v9, &v7, 4u);
  }
  sub_CC21A0((__int64)v4, &v7, 8u);
  return v7;
}
