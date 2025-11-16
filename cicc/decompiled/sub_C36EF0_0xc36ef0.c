// Function: sub_C36EF0
// Address: 0xc36ef0
//
unsigned int *__fastcall sub_C36EF0(_DWORD **a1, char a2)
{
  int v2; // eax
  unsigned int v3; // r13d
  __int64 v4; // rax

  v2 = (*a1)[4];
  if ( v2 == 2 )
    BUG();
  if ( v2 == 1 )
    return sub_C36070((__int64)a1, 0, a2, 0);
  *((_BYTE *)a1 + 20) = *((_BYTE *)a1 + 20) & 0xF0 | (8 * (a2 & 1));
  *((_DWORD *)a1 + 4) = sub_C36ED0(a1);
  v3 = sub_C337D0((__int64)a1);
  v4 = sub_C33900((__int64)a1);
  return (unsigned int *)sub_C45D00(v4, 0, v3);
}
