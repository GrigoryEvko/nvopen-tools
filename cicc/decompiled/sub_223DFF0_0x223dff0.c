// Function: sub_223DFF0
// Address: 0x223dff0
//
void __fastcall sub_223DFF0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  int v6; // esi

  v3 = *a2;
  *(_BYTE *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a2;
  v4 = (__int64)a2 + *(_QWORD *)(v3 - 24);
  if ( *(_QWORD *)(v4 + 216) )
  {
    v6 = *(_DWORD *)(v4 + 32);
    if ( v6 )
    {
LABEL_3:
      sub_222DC80(v4, v6 | 4);
      return;
    }
    sub_223DF30(*(_QWORD **)(v4 + 216));
    v4 = (__int64)a2 + *(_QWORD *)(*a2 - 24);
  }
  v6 = *(_DWORD *)(v4 + 32);
  if ( v6 )
    goto LABEL_3;
  *(_BYTE *)a1 = 1;
}
