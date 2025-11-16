// Function: sub_931270
// Address: 0x931270
//
__int64 __fastcall sub_931270(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 *v7; // rbx
  __int64 *v8; // r15
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-60h]
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 *v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 128LL);
  if ( !v2 )
    sub_91B8A0("label for goto statement not found!", (_DWORD *)a2, 1);
  sub_92FD10(a1, (unsigned int *)a2);
  sub_91CAC0((_QWORD *)a2);
  if ( *(_BYTE *)(a1 + 240) )
  {
    sub_9310E0((__int64)&v15, a1, a2);
    v6 = v17;
    v7 = v16;
    v14 = (__int64)v15;
    v8 = v15;
    v13 = v17;
    while ( v7 != v8 )
    {
      v9 = *v8++;
      sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(v9 + 120), v5, v6);
      sub_9465D0(a1, v9, 1);
    }
    v10 = *(_QWORD *)(a1 + 424);
    v11 = *(_QWORD *)(v10 - 24);
    if ( v11 != *(_QWORD *)(v10 - 16) )
      *(_QWORD *)(v10 - 16) = v11;
    v12 = sub_946C80(a1, v2);
    sub_92FD90(a1, v12);
    result = v14;
    if ( v14 )
      return j_j___libc_free_0(v14, v13 - v14);
  }
  else
  {
    v3 = sub_946C80(a1, v2);
    return sub_92FD90(a1, v3);
  }
  return result;
}
