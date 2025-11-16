// Function: sub_86F800
// Address: 0x86f800
//
__int64 __fastcall sub_86F800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 result; // rax

  v6 = 0x2E8BA2E8BA2E8BA3LL * ((qword_4D03B98 - qword_4F5FD90) >> 4);
  v7 = v6 + unk_4D03B90 + 1;
  if ( v7 == qword_4F5FD88 )
  {
    sub_86B3C0(a1, a2, v7, a4, a5, a6);
    v6 = 0x2E8BA2E8BA2E8BA3LL * ((qword_4D03B98 - qword_4F5FD90) >> 4);
  }
  *(_QWORD *)a1 = v6;
  v8 = unk_4D03B90;
  unk_4D03B90 = -1;
  *(_DWORD *)(a1 + 8) = v8;
  qword_4D03B98 += 176 * (v8 + 1);
  *(_QWORD *)(a1 + 12) = qword_4F5FD78;
  *(_DWORD *)(a1 + 20) = dword_4F5FD80;
  *(_QWORD *)(a1 + 24) = qword_4F5FD70;
  result = qword_4F5FD68;
  *(_QWORD *)(a1 + 32) = qword_4F5FD68;
  return result;
}
