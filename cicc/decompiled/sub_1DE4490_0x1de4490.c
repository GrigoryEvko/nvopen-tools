// Function: sub_1DE4490
// Address: 0x1de4490
//
__int64 __fastcall sub_1DE4490(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // r15
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 *v15; // rbx
  __int64 v16; // r14
  int v17; // eax
  _QWORD *v19; // [rsp+0h] [rbp-50h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v19 = (_QWORD *)(a2 + 320);
  if ( a2 + 320 != *(_QWORD *)(*(_QWORD *)(a2 + 328) + 8LL) )
  {
    v2 = (__int64 *)a1[1];
    v4 = *v2;
    v5 = v2[1];
    if ( v4 == v5 )
LABEL_22:
      BUG();
    while ( *(_UNKNOWN **)v4 != &unk_4FC5828 )
    {
      v4 += 16;
      if ( v5 == v4 )
        goto LABEL_22;
    }
    v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4FC5828);
    v7 = (__int64 *)a1[1];
    a1[29] = v6;
    v8 = *v7;
    v9 = v7[1];
    if ( v8 == v9 )
LABEL_21:
      BUG();
    while ( *(_UNKNOWN **)v8 != &unk_4FC453D )
    {
      v8 += 16;
      if ( v9 == v8 )
        goto LABEL_21;
    }
    v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4FC453D);
    a1[30] = v10;
    v11 = v10;
    v12 = *(_QWORD **)(a2 + 328);
    if ( v12 != v19 )
    {
      while ( 1 )
      {
        v13 = sub_1DDC3C0(v11, (__int64)v12);
        v14 = (__int64 *)v12[12];
        v15 = (__int64 *)v12[11];
        for ( v20[0] = v13; v14 != v15; ++v15 )
        {
          v16 = *v15;
          if ( !sub_1DD69A0((__int64)v12, *v15) )
          {
            v17 = sub_1DF1780(a1[29], v12, v16);
            sub_16AF500(v20, v17);
          }
        }
        v12 = (_QWORD *)v12[1];
        if ( v19 == v12 )
          break;
        v11 = a1[30];
      }
    }
  }
  return 0;
}
