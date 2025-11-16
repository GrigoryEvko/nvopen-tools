// Function: sub_29E7990
// Address: 0x29e7990
//
__int64 __fastcall sub_29E7990(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned int *v16; // rax
  _BYTE *v17; // rax
  int v19; // eax
  __int64 v20; // [rsp+0h] [rbp-50h]

  v3 = *(_QWORD *)(a1 + 56);
  v20 = a1;
  if ( v3 == a1 + 48 )
    return 0;
  while ( 1 )
  {
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 8);
    if ( *(_BYTE *)(v4 - 24) == 85
      && !(unsigned __int8)sub_A73ED0((_QWORD *)(v4 + 48), 41)
      && !(unsigned __int8)sub_B49560(v4 - 24, 41) )
    {
      v5 = *(_QWORD *)(v4 - 56);
      if ( !v5
        || *(_BYTE *)v5
        || *(_QWORD *)(v5 + 24) != *(_QWORD *)(v4 + 56)
        || (v19 = *(_DWORD *)(v5 + 36), v19 != 146) && v19 != 153 )
      {
        if ( *(char *)(v4 - 17) >= 0 )
          break;
        v6 = sub_BD2BC0(v4 - 24);
        v11 = v6 + v10;
        if ( *(char *)(v4 - 17) < 0 )
          v11 -= sub_BD2BC0(v4 - 24);
        v12 = v11 >> 4;
        if ( !(_DWORD)v12 )
          break;
        v13 = 0;
        v14 = 16LL * (unsigned int)v12;
        while ( 1 )
        {
          v15 = 0;
          if ( *(char *)(v4 - 17) < 0 )
            v15 = sub_BD2BC0(v4 - 24);
          v16 = (unsigned int *)(v13 + v15);
          if ( *(_DWORD *)(*(_QWORD *)v16 + 8LL) == 1 )
            break;
          v13 += 16;
          if ( v13 == v14 )
            goto LABEL_19;
        }
        v17 = sub_29E73A0(
                *(unsigned __int8 **)(v4 - 24 + 32 * (v16[2] - (unsigned __int64)(*(_DWORD *)(v4 - 20) & 0x7FFFFFF))),
                a3,
                *(_DWORD *)(v4 - 20) & 0x7FFFFFF,
                v7,
                v8,
                v9);
        if ( !v17 || *v17 == 21 )
          break;
      }
    }
    if ( a1 + 48 == v3 )
      return 0;
  }
LABEL_19:
  sub_F566B0((unsigned __int8 *)(v4 - 24), a2, 0);
  return v20;
}
