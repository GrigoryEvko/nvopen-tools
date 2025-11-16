// Function: sub_1DD64C0
// Address: 0x1dd64c0
//
__int64 __fastcall sub_1DD64C0(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rdx
  __int64 v4; // rax

  v2 = *(_DWORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v2 <= 3u )
  {
    v4 = sub_16E7EE0(a2, "%bb.", 4u);
    return sub_16E7AB0(v4, *(int *)(a1 + 48));
  }
  else
  {
    *v2 = 778199589;
    *(_QWORD *)(a2 + 24) += 4LL;
    return sub_16E7AB0(a2, *(int *)(a1 + 48));
  }
}
