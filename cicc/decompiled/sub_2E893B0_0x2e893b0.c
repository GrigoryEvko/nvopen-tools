// Function: sub_2E893B0
// Address: 0x2e893b0
//
__int64 __fastcall sub_2E893B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int16 v4; // dx
  _QWORD *v5; // rdi
  __int64 *v6; // r15
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rcx
  _BYTE *v12; // rax
  __int64 v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = sub_2E891C0(a1);
  v4 = *(_WORD *)(a1 + 68);
  v5 = (_QWORD *)v3;
  if ( v4 != 14 )
  {
    if ( v4 == 15 )
    {
      v6 = *(__int64 **)a2;
      v7 = *(unsigned int *)(a2 + 8);
      v13[0] = 6;
      v8 = &v6[v7];
      if ( v8 != v6 )
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(a1 + 32);
          if ( v4 != 14 )
            v9 += 80;
          v10 = *v6++;
          v5 = (_QWORD *)sub_B0DBA0(v5, v13, 1, -858993459 * (unsigned int)((v10 - v9) >> 3), 0);
          if ( v8 == v6 )
            break;
          v4 = *(_WORD *)(a1 + 68);
        }
      }
    }
    return (__int64)v5;
  }
  v12 = *(_BYTE **)(a1 + 32);
  if ( v12[40] != 1 || *v12 )
    return (__int64)v5;
  return sub_B0DAC0(v5, 1, 0);
}
