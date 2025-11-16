// Function: sub_F0C930
// Address: 0xf0c930
//
__int64 __fastcall sub_F0C930(__int64 a1, __int64 a2)
{
  char v3; // dl
  char v4; // dl
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  char v8; // dl
  char v9; // dl
  __int64 v10; // [rsp-38h] [rbp-38h]
  __int64 v11; // [rsp-38h] [rbp-38h]
  char v12; // [rsp-30h] [rbp-30h]
  char v13; // [rsp-30h] [rbp-30h]

  if ( *(_BYTE *)a2 != 77 )
    return 0;
  v10 = sub_9208B0(*(_QWORD *)(a1 + 88), *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL));
  v12 = v3;
  if ( v10 != sub_9208B0(*(_QWORD *)(a1 + 88), *(_QWORD *)(a2 + 8)) )
    return 0;
  if ( v12 != v4 )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v5 != 76 )
    return 0;
  v6 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v7 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  if ( *(_DWORD *)(v7 + 8) >> 8 == *(_DWORD *)(v6 + 8) >> 8
    && (v11 = sub_9208B0(*(_QWORD *)(a1 + 88), *(_QWORD *)(v5 + 8)),
        v13 = v8,
        sub_9208B0(*(_QWORD *)(a1 + 88), *(_QWORD *)(*(_QWORD *)(v5 - 32) + 8LL)) == v11)
    && v9 == v13 )
  {
    return *(_QWORD *)(v5 - 32);
  }
  else
  {
    return 0;
  }
}
