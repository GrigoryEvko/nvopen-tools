// Function: sub_2F40450
// Address: 0x2f40450
//
__int64 __fastcall sub_2F40450(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 (*v9)(); // rdx
  int v10; // eax
  unsigned int v11; // edx
  __int64 result; // rax

  v3 = a3 + 6;
  *(_QWORD *)a1 = &unk_4A2AC10;
  v4 = a3[5];
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = v4;
  v5 = a3[4];
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 32) = v5;
  v6 = (_QWORD *)a3[3];
  *(_QWORD *)(a1 + 40) = v6;
  *(_QWORD *)(a1 + 48) = *v6;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 64) = v3;
  *(_QWORD *)(a1 + 56) = v7;
  v8 = v7;
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 328LL);
  v10 = 0;
  if ( v9 != sub_2F3F790 )
    v10 = ((__int64 (__fastcall *)(__int64, __int64))v9)(v8, a2);
  v11 = *(_DWORD *)(v8 + 16);
  *(_QWORD *)(a1 + 72) = **(_QWORD **)(v8 + 248) + v11 * v10;
  *(_QWORD *)(a1 + 80) = v11;
  result = (unsigned __int8)qword_50235A8;
  if ( !(_BYTE)qword_50235A8 )
    result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 16) + 400LL))(
               *(_QWORD *)(a2 + 16),
               *(unsigned int *)(*(_QWORD *)(a2 + 8) + 648LL));
  *(_BYTE *)(a1 + 88) = result;
  return result;
}
