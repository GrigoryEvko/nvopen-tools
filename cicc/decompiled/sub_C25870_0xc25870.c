// Function: sub_C25870
// Address: 0xc25870
//
__int64 __fastcall sub_C25870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6, __int64 a7, __int64 a8)
{
  __int64 v11; // r9
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // [rsp-8h] [rbp-88h]
  __int64 v16[2]; // [rsp+0h] [rbp-80h] BYREF
  char v17; // [rsp+10h] [rbp-70h]
  _QWORD v18[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v18[0] = a2;
  v18[1] = a3;
  v19 = 261;
  sub_C1F520((__int64)v16, (__int64)v18, a5);
  if ( (v17 & 1) != 0 && (v13 = v16[0], v14 = v16[1], LODWORD(v16[0])) )
  {
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v13;
    *(_QWORD *)(a1 + 8) = v14;
  }
  else
  {
    sub_C24C60(a1, v16, a4, a5, a6, v11, a7, a8);
    if ( (v17 & 1) == 0 && v16[0] )
      (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v16[0] + 8LL))(v16[0], v16, v15);
  }
  return a1;
}
