// Function: sub_921E00
// Address: 0x921e00
//
__int64 __fastcall sub_921E00(__int64 a1, __int64 a2)
{
  int v2; // r15d
  unsigned __int8 v3; // bl
  int v4; // eax

  if ( sub_91B770(*(_QWORD *)a2) )
    sub_91B8A0("cannot evaluate expression with aggregate type as bool!", (_DWORD *)(a2 + 36), 1);
  v2 = sub_BCB2A0(*(_QWORD *)(a1 + 40));
  v3 = sub_91B6F0(*(_QWORD *)a2);
  v4 = sub_92F410(a1, a2);
  return sub_92C9E0(a1, v4, v3, v2, 0, 0, a2 + 36);
}
