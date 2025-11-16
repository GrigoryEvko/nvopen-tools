// Function: sub_1F6DBA0
// Address: 0x1f6dba0
//
bool __fastcall sub_1F6DBA0(unsigned int *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // r12
  unsigned int v4; // ebx
  unsigned __int64 v5; // rax
  bool result; // al
  unsigned int v7; // ebx

  v2 = *a1;
  v3 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD *)(v3 + 24);
    return v2 <= v5;
  }
  v7 = v4 - sub_16A57B0(v3 + 24);
  result = 1;
  if ( v7 <= 0x40 )
  {
    v5 = **(_QWORD **)(v3 + 24);
    return v2 <= v5;
  }
  return result;
}
