// Function: sub_2D56C30
// Address: 0x2d56c30
//
__int64 __fastcall sub_2D56C30(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v2 = *(_QWORD *)(a2 + 16);
  if ( v2 && !*(_QWORD *)(v2 + 8) && sub_991A70((unsigned __int8 *)a2, 0, 0, 0, 0, 1u, 0) )
    return sub_DFCD70(a1, a2);
  else
    return 0;
}
