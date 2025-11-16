// Function: sub_2DEA820
// Address: 0x2dea820
//
__int64 __fastcall sub_2DEA820(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  size_t v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 (*v18)(); // rdx
  int v19; // eax

  v3 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190);
  if ( !v3 )
    return 0;
  v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_5027190);
  if ( !v4 || !(_BYTE)qword_501E7E8 )
    return 0;
  v6 = *(__int64 **)(a1 + 8);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
    goto LABEL_17;
  while ( *(_UNKNOWN **)v7 != &unk_4F8144C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_17;
  }
  *(_QWORD *)(a1 + 176) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
                            *(_QWORD *)(v7 + 8),
                            &unk_4F8144C)
                        + 176;
  v9 = *(_QWORD *)(v4 + 256);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 16LL);
  if ( v10 == sub_23CE270 )
LABEL_17:
    BUG();
  v11 = ((__int64 (__fastcall *)(__int64, __int64))v10)(v9, a2);
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 144LL);
  if ( v12 == sub_2C8F680 )
  {
    *(_QWORD *)(a1 + 184) = 0;
    goto LABEL_17;
  }
  v13 = ((__int64 (__fastcall *)(__int64))v12)(v11);
  *(_QWORD *)(a1 + 184) = v13;
  v17 = v13;
  v18 = *(__int64 (**)())(*(_QWORD *)v13 + 1504LL);
  v19 = 2;
  if ( v18 != sub_2DE6890 )
    v19 = ((__int64 (__fastcall *)(__int64))v18)(v17);
  *(_DWORD *)(a1 + 192) = v19;
  return sub_2DE7CB0(a1 + 176, a2, (__int64)v18, v14, v15, v16);
}
