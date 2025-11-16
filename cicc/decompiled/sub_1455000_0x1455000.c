// Function: sub_1455000
// Address: 0x1455000
//
bool __fastcall sub_1455000(__int64 a1)
{
  unsigned int v1; // ebx

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return *(_QWORD *)a1 == 1;
  else
    return v1 - 1 == (unsigned int)sub_16A57B0(a1);
}
