// Function: sub_234BC60
// Address: 0x234bc60
//
__int64 __fastcall sub_234BC60(__int64 a1, __int64 *a2, char a3)
{
  char v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rax

  v4 = *((_BYTE *)a2 + 8);
  v5 = *a2;
  v6 = sub_22077B0(0x18u);
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = v5;
    *(_BYTE *)(v6 + 16) = v4;
    *(_QWORD *)v6 = &unk_4A10738;
  }
  *(_QWORD *)a1 = v6;
  *(_BYTE *)(a1 + 8) = a3;
  return a1;
}
