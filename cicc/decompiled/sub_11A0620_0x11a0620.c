// Function: sub_11A0620
// Address: 0x11a0620
//
__int64 __fastcall sub_11A0620(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 48) = 0;
  if ( v2 != a1 + 24 )
    return _libc_free(v2, a2);
  return result;
}
