// Function: sub_134C930
// Address: 0x134c930
//
__int64 __fastcall sub_134C930(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax
  __int64 *v9; // rbx
  __int64 *v10; // r13
  __int64 v12; // rdi

  *(_WORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v2 = a2[5];
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = v2;
  v3 = a2[6];
  a2[6] = 0;
  *(_QWORD *)(a1 + 48) = v3;
  v4 = a2[7];
  a2[7] = 0;
  *(_QWORD *)(a1 + 56) = v4;
  v5 = a2[8];
  a2[8] = 0;
  *(_QWORD *)(a1 + 64) = v5;
  v6 = a2[9];
  a2[9] = 0;
  *(_QWORD *)(a1 + 72) = v6;
  v7 = a2[10];
  a2[10] = 0;
  *(_QWORD *)(a1 + 80) = v7;
  result = a2[11];
  a2[11] = 0;
  v9 = *(__int64 **)(a1 + 48);
  v10 = *(__int64 **)(a1 + 56);
  for ( *(_QWORD *)(a1 + 88) = result;
        v10 != v9;
        result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 16LL))(v12, a1) )
  {
    v12 = *v9++;
  }
  return result;
}
