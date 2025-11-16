// Function: sub_3754A80
// Address: 0x3754a80
//
__int64 __fastcall sub_3754A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 (*v12)(); // rcx
  __int64 (*v13)(void); // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 result; // rax

  v7 = *(_QWORD *)(a3 + 32);
  v8 = *(_QWORD *)(v7 + 32);
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 8) = v8;
  v9 = *(__int64 **)(v7 + 16);
  v10 = 0;
  v11 = *v9;
  v12 = *(__int64 (**)())(*v9 + 128);
  if ( v12 != sub_2DAC790 )
  {
    v10 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))v12)(v9, a2, 0);
    v11 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
  }
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 24) = (*(__int64 (**)(void))(v11 + 200))();
  v13 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 144LL);
  v14 = 0;
  if ( v13 != sub_2C8F680 )
    v14 = v13();
  *(_QWORD *)(a1 + 40) = a3;
  v15 = *(_QWORD *)(a3 + 32);
  *(_QWORD *)(a1 + 48) = a4;
  *(_QWORD *)(a1 + 32) = v14;
  result = sub_2E799E0(v15);
  *(_BYTE *)(a1 + 56) = result;
  return result;
}
