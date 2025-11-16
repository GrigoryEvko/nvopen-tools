// Function: sub_89F0F0
// Address: 0x89f0f0
//
__int64 __fastcall sub_89F0F0(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax
  _QWORD *v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r12

  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14) & 4) == 0 )
    sub_89EFB0(a2, a1);
  v2 = *(_QWORD *)(a2 + 192);
  sub_897CB0(a2, a1);
  *(_DWORD *)(v2 + 8) = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  v3 = sub_727300();
  v3[3] = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
  v3[4] = *(_QWORD *)&dword_4F077C8;
  *v3 = *(_QWORD *)(a2 + 488);
  v4 = *(_QWORD *)(a2 + 192);
  *(_QWORD *)(a2 + 488) = v3;
  if ( v4 )
    *(_QWORD *)(v4 + 32) = v3;
  *(_DWORD *)(a2 + 280) = dword_4F06650[0];
  result = sub_89D5D0(a2, a1);
  v6 = *(_QWORD **)v2;
  v7 = *(_QWORD *)(v2 + 32);
  if ( v6 )
  {
    result = 0;
    do
    {
      while ( 1 )
      {
        v8 = result;
        result = (__int64)sub_8992B0((__int64)v6);
        if ( !v8 )
          break;
        *(_QWORD *)(v8 + 112) = result;
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          return result;
      }
      *(_QWORD *)(v7 + 8) = result;
      v6 = (_QWORD *)*v6;
    }
    while ( v6 );
  }
  return result;
}
