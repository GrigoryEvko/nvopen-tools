// Function: sub_2F524F0
// Address: 0x2f524f0
//
bool __fastcall sub_2F524F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  char v7; // al
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // [rsp+8h] [rbp-28h] BYREF
  __int64 v15; // [rsp+10h] [rbp-20h] BYREF
  __int64 v16; // [rsp+18h] [rbp-18h]

  v2 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)a2 = 0;
  *(_QWORD *)(a2 + 16) = 0;
  if ( v2 )
    --*(_DWORD *)(v2 + 8);
  *(_QWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a2 + 88) = 0;
  *(_DWORD *)(a2 + 32) = 0;
  *(_DWORD *)(a2 + 104) = 0;
  sub_2FB0010(a1[104], a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v14 = 0;
  v16 = 0;
  v15 = v6;
  if ( v6 )
    ++*(_DWORD *)(v6 + 8);
  v7 = sub_2F51220(a1, &v15, &v14, v3, v4, v5);
  v10 = v15;
  v16 = 0;
  if ( v15 )
    --*(_DWORD *)(v15 + 8);
  if ( !v7 || !(unsigned __int8)sub_2F51D00((__int64)a1, a2, v10, v8, v9) )
    return 0;
  sub_2FB0100(a1[104]);
  v12 = sub_2F4C690(*(_QWORD **)(a2 + 24), *(_QWORD *)(a2 + 24) + 8LL * *(unsigned int *)(a2 + 32));
  return v13 != v12;
}
