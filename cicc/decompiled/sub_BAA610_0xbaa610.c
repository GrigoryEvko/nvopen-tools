// Function: sub_BAA610
// Address: 0xbaa610
//
__int64 __fastcall sub_BAA610(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // rax
  __int64 v5; // [rsp+8h] [rbp-8h]

  v1 = sub_BA91D0(a1, "Code Model", 0xAu);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 136);
    v3 = *(_QWORD **)(v2 + 24);
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
      v3 = (_QWORD *)*v3;
    LODWORD(v5) = (_DWORD)v3;
    BYTE4(v5) = 1;
    return v5;
  }
  else
  {
    BYTE4(v5) = 0;
    return v5;
  }
}
