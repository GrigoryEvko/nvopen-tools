// Function: sub_C37310
// Address: 0xc37310
//
__int64 __fastcall sub_C37310(__int64 a1, char a2)
{
  __int64 v2; // rdx
  char v3; // si
  unsigned int v4; // r13d
  __int64 v5; // rax

  v2 = *(_QWORD *)a1;
  if ( !*(_BYTE *)(*(_QWORD *)a1 + 24LL) )
    BUG();
  v3 = *(_BYTE *)(a1 + 20) & 0xF0 | (8 * a2 + 3) & 0xF;
  *(_BYTE *)(a1 + 20) = v3;
  if ( *(_DWORD *)(v2 + 20) == 2 )
    *(_BYTE *)(a1 + 20) = v3 & 0xF7;
  *(_DWORD *)(a1 + 16) = sub_C36EE0(a1);
  v4 = sub_C337D0(a1);
  v5 = sub_C33900(a1);
  return sub_C45D00(v5, 0, v4);
}
