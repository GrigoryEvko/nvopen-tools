// Function: sub_38BD490
// Address: 0x38bd490
//
__int64 __fastcall sub_38BD490(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned int v5; // eax
  __int64 result; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rcx
  __int64 v12; // [rsp+8h] [rbp-18h]

  LOBYTE(v3) = a3;
  v4 = *(_QWORD *)(a1 + 32);
  if ( !v4 )
    goto LABEL_14;
  v5 = *(_DWORD *)(v4 + 680);
  if ( v5 == 2 )
  {
    result = sub_38E22B0(40, a2, a1);
    if ( !result )
      return 0;
    v11 = *(_QWORD *)(result + 8) & 0xFFFF0000FFFE0000LL;
    *(_DWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_QWORD *)(result + 8) = v11 | (unsigned __int8)v3 | 0x40LL;
    *(_QWORD *)result = 4LL * (a2 != 0);
    if ( a2 )
      *(_QWORD *)(result - 8) = a2;
    *(_WORD *)(result + 32) = 0;
    return result;
  }
  if ( v5 <= 2 )
  {
    if ( v5 )
    {
      result = sub_38E22B0(40, a2, a1);
      if ( result )
      {
        v7 = *(_QWORD *)(result + 8) & 0xFFFF0000FFFE0000LL;
        *(_DWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 8) = v7 | (unsigned __int8)v3 | 0x80u;
        *(_QWORD *)result = 4LL * (a2 != 0);
        if ( a2 )
          *(_QWORD *)(result - 8) = a2;
        *(_QWORD *)(result + 32) = 0;
        return result;
      }
      return 0;
    }
    result = sub_38E22B0(32, a2, a1);
    if ( !result )
      return 0;
    v10 = *(_QWORD *)(result + 8) & 0xFFFF0000FFFE0000LL | (unsigned __int8)v3 | 0xC0u;
    goto LABEL_16;
  }
  if ( v5 != 3 )
  {
LABEL_14:
    result = sub_38E22B0(32, a2, a1);
    if ( !result )
      return 0;
    v10 = *(_QWORD *)(result + 8) & 0xFFFF0000FFFE0000LL | (unsigned __int8)v3;
LABEL_16:
    *(_QWORD *)(result + 8) = v10;
    *(_DWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_QWORD *)result = 4LL * (a2 != 0);
    if ( a2 )
      *(_QWORD *)(result - 8) = a2;
    return result;
  }
  v8 = sub_38E22B0(144, a2, a1);
  if ( !v8 )
    return 0;
  v3 = (unsigned __int8)v3;
  v9 = *(_QWORD *)(v8 + 8) & 0xFFFF0000FFFE0000LL;
  BYTE1(v3) = 1;
  *(_DWORD *)(v8 + 16) = 0;
  *(_QWORD *)(v8 + 24) = 0;
  *(_QWORD *)(v8 + 8) = v9 | v3;
  *(_QWORD *)v8 = 4LL * (a2 != 0);
  if ( a2 )
    *(_QWORD *)(v8 - 8) = a2;
  *(_DWORD *)(v8 + 32) = 1;
  *(_WORD *)(v8 + 36) = 0;
  *(_QWORD *)(v8 + 40) = v8 + 56;
  *(_BYTE *)(v8 + 38) = 0;
  v12 = v8;
  sub_38BB9D0((__int64 *)(v8 + 40), "env", (__int64)"");
  *(_QWORD *)(v12 + 80) = 0x100000000LL;
  *(_QWORD *)(v12 + 72) = v12 + 88;
  *(_QWORD *)(v12 + 104) = 0x400000000LL;
  *(_QWORD *)(v12 + 96) = v12 + 112;
  *(_WORD *)(v12 + 130) = 0;
  *(_BYTE *)(v12 + 132) = 0;
  *(_QWORD *)(v12 + 136) = 0;
  return v12;
}
