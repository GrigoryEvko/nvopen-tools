// Function: sub_1B42DA0
// Address: 0x1b42da0
//
__int64 __fastcall sub_1B42DA0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx

  if ( !(unsigned __int8)sub_1593E50(a1)
    && !(unsigned __int8)sub_1593E60(a1)
    && ((v3 = *(unsigned __int8 *)(a1 + 16), (_BYTE)v3 == 14)
     || (unsigned __int8)v3 <= 0xFu
     && ((v4 = 41504, _bittest64(&v4, v3)) || (unsigned __int8)v3 <= 3u)
     && ((_BYTE)v3 != 5
      || (unsigned __int8)sub_1594530(a1)
      && (unsigned __int8)sub_1B42DA0(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)), a2))) )
  {
    return sub_14A2DE0(a2);
  }
  else
  {
    return 0;
  }
}
