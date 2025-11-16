// Function: sub_19531C0
// Address: 0x19531c0
//
__int64 __fastcall sub_19531C0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  _QWORD *v3; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned int v8; // r14d
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  _QWORD *v14; // rax
  unsigned __int8 v15; // r8
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  unsigned __int8 v19; // [rsp+18h] [rbp-48h]
  unsigned __int8 v20; // [rsp+18h] [rbp-48h]
  bool v21; // [rsp+2Eh] [rbp-32h] BYREF
  unsigned __int8 v22; // [rsp+2Fh] [rbp-31h]

  v2 = sub_157EBA0(a2);
  if ( *(_BYTE *)(v2 + 16) == 26 )
  {
    v3 = (_QWORD *)v2;
    if ( (*(_DWORD *)(v2 + 20) & 0xFFFFFFF) == 3 )
    {
      v16 = *(_QWORD *)(v2 - 72);
      v6 = sub_157F0B0(a2);
      v7 = sub_157EB90(a2);
      v17 = sub_1632FA0(v7);
      if ( v6 )
      {
        v8 = 0;
        v18 = a2;
        if ( dword_4FB01C0 )
        {
          while ( 1 )
          {
            v9 = sub_157EBA0(v6);
            if ( *(_BYTE *)(v9 + 16) != 26 )
              break;
            if ( (*(_DWORD *)(v9 + 20) & 0xFFFFFFF) != 3 )
              break;
            v10 = *(_QWORD *)(v9 - 24);
            if ( v10 != v18 && *(_QWORD *)(v9 - 48) != v18 )
              break;
            sub_14BCF40(&v21, *(_QWORD *)(v9 - 72), v16, v17, v10 == v18, 0);
            if ( v22 )
            {
              v19 = v22;
              v12 = *(_QWORD *)((char *)v3 - 24 - 24LL * v21);
              v13 = *(_QWORD *)((char *)v3 - 24 - 24LL * !v21);
              sub_157F2D0(v12, a2, 0);
              v14 = sub_1648A60(56, 1u);
              v15 = v19;
              if ( v14 )
              {
                sub_15F8320((__int64)v14, v13, (__int64)v3);
                v15 = v19;
              }
              v20 = v15;
              sub_15F20C0(v3);
              sub_15CDBF0(*(_QWORD *)(a1 + 24), a2, v12);
              return v20;
            }
            ++v8;
            v11 = sub_157F0B0(v6);
            if ( v11 )
            {
              v18 = v6;
              v6 = v11;
              if ( dword_4FB01C0 > v8 )
                continue;
            }
            break;
          }
        }
      }
    }
  }
  return 0;
}
