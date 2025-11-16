// Function: sub_2F5F430
// Address: 0x2f5f430
//
_QWORD *__fastcall sub_2F5F430(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // rdi
  _QWORD *v12; // rax
  __int64 v13; // rax

  v9 = (_QWORD *)sub_22077B0(0x48u);
  v10 = v9;
  if ( v9 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    v9[1] = a4;
    *v9 = &unk_4A2B2B0;
    v9[2] = *(_QWORD *)(a4 + 32);
    v12 = *(_QWORD **)(a4 + 24);
    v10[3] = v12;
    v10[4] = *v12;
    v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 200LL))(v11);
    v10[7] = a5;
    v10[5] = v13;
    v10[6] = a4 + 48;
    *((_WORD *)v10 + 32) = *(_WORD *)(a4 + 29080);
    *v10 = &unk_4A2B260;
  }
  *a1 = v10;
  return a1;
}
