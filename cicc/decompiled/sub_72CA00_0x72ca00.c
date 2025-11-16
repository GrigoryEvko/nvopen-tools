// Function: sub_72CA00
// Address: 0x72ca00
//
_BYTE *__fastcall sub_72CA00(__int64 a1, void *a2)
{
  __int64 v3; // rax
  _BYTE *result; // rax
  __int64 v5; // rbx
  __int64 v6; // rdi

  v3 = sub_879C70(a2);
  if ( !v3
    || *(_BYTE *)(v3 + 80) != 9
    || (v5 = *(_QWORD *)(v3 + 88)) == 0
    || *(_BYTE *)(v5 + 177) != 1
    || (v6 = *(_QWORD *)(v5 + 120), (*(_BYTE *)(v6 + 140) & 0xFB) != 8)
    || (sub_8D4C10(v6, dword_4F077C4 != 2) & 1) == 0
    || (result = *(_BYTE **)(v5 + 184)) == 0 )
  {
    sub_686490(0xBACu, dword_4F07508, a1, (__int64)a2);
    return sub_72C9A0();
  }
  return result;
}
