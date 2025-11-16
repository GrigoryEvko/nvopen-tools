// Function: sub_139D370
// Address: 0x139d370
//
__int64 __fastcall sub_139D370(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax

  a1[20] = a2;
  v3 = sub_160F9A0(a1[1], &unk_4F9D3C0, 1);
  if ( v3 && (v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F9D3C0)) != 0 )
    v5 = sub_14A4050(v4, a2);
  else
    v5 = 0;
  a1[21] = v5;
  return 0;
}
