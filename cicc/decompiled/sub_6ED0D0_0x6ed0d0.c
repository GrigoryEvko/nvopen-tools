// Function: sub_6ED0D0
// Address: 0x6ed0d0
//
__int64 __fastcall sub_6ED0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v11; // r12
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v8 = a1 + 144;
  v12[0] = v7;
  if ( (*(_BYTE *)(a1 + 20) & 4) != 0 && *(_BYTE *)(a1 + 317) == 7 )
  {
    v11 = *(_QWORD *)(a1 + 344);
    if ( v11 )
    {
      sub_72A510(v8, v7);
      v8 = v12[0];
      *(_QWORD *)(v12[0] + 184) = sub_7CADA0(v11, a1 + 24);
    }
  }
  v9 = sub_73A720(v8);
  *(_QWORD *)(v9 + 36) = *(_QWORD *)(a1 + 68);
  *(_QWORD *)(v9 + 44) = *(_QWORD *)(a1 + 76);
  *(_QWORD *)(v9 + 28) = *(_QWORD *)(a1 + 68);
  if ( *(_BYTE *)(a1 + 17) == 1 && !sub_6ED0A0(a1) )
  {
    *(_BYTE *)(v9 + 25) |= 1u;
    *(_QWORD *)v9 = *(_QWORD *)a1;
  }
  sub_724E30(v12);
  return v9;
}
