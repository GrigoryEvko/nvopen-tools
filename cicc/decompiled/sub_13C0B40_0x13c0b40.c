// Function: sub_13C0B40
// Address: 0x13c0b40
//
__int64 __fastcall sub_13C0B40(__int64 a1)
{
  void (*v1)(void); // rax
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 96LL);
  if ( (char *)v1 == (char *)sub_13BF5B0 )
  {
    sub_13BF460(*(_QWORD **)(a1 + 176));
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 184) = a1 + 168;
    *(_QWORD *)(a1 + 192) = a1 + 168;
    *(_QWORD *)(a1 + 200) = 0;
  }
  else
  {
    v1();
  }
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_9:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_9;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  sub_13C0A40(a1 + 160, v5 + 160);
  return 0;
}
