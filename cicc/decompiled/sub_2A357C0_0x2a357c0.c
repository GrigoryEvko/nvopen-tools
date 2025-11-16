// Function: sub_2A357C0
// Address: 0x2a357c0
//
__int64 __fastcall sub_2A357C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 v8; // rax
  __m128i *v9; // r8
  __m128i *v10; // r9
  __int64 v11; // rdi
  __int64 v12; // rdx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FDBCD4 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_10;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FDBCD4);
  v7 = (__int64 *)sub_22C1470(v6);
  v8 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8662C);
  if ( v8 && (v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4F8662C)) != 0 )
    v12 = sub_CFFAC0(v11, a2);
  else
    v12 = 0;
  return sub_2A33580(a2, v7, v12, *(_BYTE *)(a1 + 169), v9, v10);
}
