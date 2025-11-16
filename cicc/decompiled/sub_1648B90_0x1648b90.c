// Function: sub_1648B90
// Address: 0x1648b90
//
__int64 __fastcall sub_1648B90(__int64 a1)
{
  char v1; // dl
  __int64 v2; // rsi
  _QWORD *v3; // r13

  v1 = *(_BYTE *)(a1 + 23);
  v2 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (v1 & 0x40) != 0 )
  {
    sub_1648630(*(_QWORD **)(a1 - 8), (_QWORD *)(*(_QWORD *)(a1 - 8) + v2), 1);
    return j___libc_free_0(a1 - 8);
  }
  else
  {
    v3 = (_QWORD *)(a1 - v2);
    if ( v1 < 0 )
    {
      sub_1648630(v3, (_QWORD *)a1, 0);
      return j___libc_free_0((char *)v3 - *(v3 - 1) - 8);
    }
    else
    {
      sub_1648630(v3, (_QWORD *)a1, 0);
      return j___libc_free_0(v3);
    }
  }
}
