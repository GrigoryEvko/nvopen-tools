// Function: sub_35DDD70
// Address: 0x35ddd70
//
__int64 __fastcall sub_35DDD70(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a2;
  v3 = sub_B2D620(*a2, "frame-pointer", 0xDu);
  if ( !(_BYTE)v3 )
    return v3;
  v7[0] = sub_B2D7E0(v2, "frame-pointer", 0xDu);
  v5 = sub_A72240(v7);
  if ( v6 == 3 )
  {
    if ( *(_WORD *)v5 == 27745 && *(_BYTE *)(v5 + 2) == 108 )
      return v3;
LABEL_14:
    BUG();
  }
  if ( v6 != 8 )
  {
    if ( v6 != 4 || *(_DWORD *)v5 != 1701736302 )
      goto LABEL_14;
    return 0;
  }
  if ( *(_QWORD *)v5 != 0x6661656C2D6E6F6ELL )
  {
    if ( *(_QWORD *)v5 != 0x6465767265736572LL )
      goto LABEL_14;
    return 0;
  }
  return *(unsigned __int8 *)(a2[6] + 66);
}
