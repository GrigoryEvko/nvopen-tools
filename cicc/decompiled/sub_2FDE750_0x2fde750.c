// Function: sub_2FDE750
// Address: 0x2fde750
//
char __fastcall sub_2FDE750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 v9; // r13
  __int64 v10; // rax

  v5 = *(_DWORD *)(a2 + 44);
  if ( (v5 & 4) == 0 && (v5 & 8) != 0 )
    LOBYTE(v6) = sub_2E88A90(a2, 512, 1);
  else
    v6 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 9) & 1LL;
  if ( !(_BYTE)v6 )
  {
    LOBYTE(v6) = 1;
    if ( (unsigned __int16)(*(_WORD *)(a2 + 68) - 2) > 4u )
    {
      v7 = *(_QWORD *)(a4 + 16);
      v8 = *(__int64 (**)())(*(_QWORD *)v7 + 144LL);
      if ( v8 == sub_2C8F680 )
      {
        (*(void (**)(void))(*(_QWORD *)v7 + 200LL))();
        BUG();
      }
      v9 = v8();
      v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a4 + 16) + 200LL))(*(_QWORD *)(a4 + 16));
      LOBYTE(v6) = (unsigned int)sub_2E8E710(a2, *(_DWORD *)(v9 + 104), v10, 0, 1) != -1;
    }
  }
  return v6;
}
