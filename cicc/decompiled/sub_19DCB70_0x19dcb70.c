// Function: sub_19DCB70
// Address: 0x19dcb70
//
__int64 __fastcall sub_19DCB70(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned int v11; // r12d
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r8
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v11 = 0;
  v12 = a2 + 72;
  *a1 = a3;
  v13 = *(_QWORD *)(a2 + 80);
  if ( v13 != a2 + 72 )
  {
    do
    {
      while ( 1 )
      {
        v14 = v13;
        v13 = *(_QWORD *)(v13 + 8);
        v15 = v14 - 24;
        v16 = sub_157EBA0(v14 - 24);
        if ( *(_BYTE *)(v16 + 16) == 26 && (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) == 3 )
        {
          v26 = *(_QWORD *)(v16 - 48);
          v24 = *(_QWORD *)(v16 - 24);
          if ( sub_157F0B0(v24) )
          {
            if ( sub_157F0B0(v26) )
            {
              v25 = sub_157F1C0(v24);
              v17 = sub_157F1C0(v26);
              if ( v17 != 0 && v25 != 0 && v25 == v17 )
                break;
            }
          }
        }
        if ( v12 == v13 )
          return v11;
      }
      v19 = sub_157EBA0(v15);
      v20 = sub_15F4DF0(v19, 0);
      v21 = sub_157F1C0(v20);
      v11 |= sub_19DBD20(a1, v21, a4, a5, a6, a7, v22, v23, a10, a11);
    }
    while ( v12 != v13 );
  }
  return v11;
}
