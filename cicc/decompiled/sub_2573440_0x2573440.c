// Function: sub_2573440
// Address: 0x2573440
//
__int64 *__fastcall sub_2573440(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int v4; // esi
  int v5; // eax
  int v6; // eax
  __int64 *result; // rax
  __int64 v8; // rdx
  bool v9; // zf
  __int64 *v10; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v10 = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= 3 * v4 )
  {
    v4 *= 2;
    goto LABEL_13;
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v6 <= v4 >> 3 )
  {
LABEL_13:
    sub_2572F80(a1, v4);
    sub_2567AA0(a1, a2, &v10);
    v6 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v6;
  if ( !byte_4FEF2C0[0] && (unsigned int)sub_2207590((__int64)byte_4FEF2C0) )
  {
    qword_4FEF2E0 = -4096;
    unk_4FEF2E8 = -4096;
    qword_4FEF2F0 = 0;
    unk_4FEF2F8 = 0;
    sub_2207640((__int64)byte_4FEF2C0);
  }
  result = v10;
  v8 = *v10;
  if ( unk_4FEF2E8 != *(_QWORD *)(*v10 + 8)
    || qword_4FEF2E0 != *(_QWORD *)v8
    || (v9 = sub_254C7C0(*(__int64 **)(v8 + 16), qword_4FEF2F0) == 0, result = v10, v9) )
  {
    --*(_DWORD *)(a1 + 20);
  }
  return result;
}
