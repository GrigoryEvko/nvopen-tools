// Function: sub_E25410
// Address: 0xe25410
//
unsigned __int64 __fastcall sub_E25410(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // r12
  _BYTE *v4; // rdx
  __int64 *v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax

  ++a2[1];
  --*a2;
  v2 = *(_QWORD **)(a1 + 16);
  v3 = (*v2 + v2[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v2[1] = v3 - *v2 + 24;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v6 = (__int64 *)sub_22077B0(32);
    v7 = v6;
    if ( v6 )
    {
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      v6[3] = 0;
    }
    v8 = sub_2207820(4096);
    v7[2] = 4096;
    *v7 = v8;
    v3 = v8;
    v9 = *(_QWORD *)(a1 + 16);
    v7[1] = 24;
    v7[3] = v9;
    *(_QWORD *)(a1 + 16) = v7;
  }
  if ( !v3 )
  {
    MEMORY[0x10] = sub_E253C0(a1, a2, 1);
    BUG();
  }
  *(_DWORD *)(v3 + 8) = 17;
  *(_BYTE *)(v3 + 12) = 0;
  *(_QWORD *)(v3 + 16) = 0;
  *(_QWORD *)v3 = &unk_49E1208;
  *(_QWORD *)(v3 + 16) = sub_E253C0(a1, a2, 1);
  v4 = (_BYTE *)a2[1];
  if ( *a2 && *v4 == 64 )
  {
    --*a2;
    a2[1] = (__int64)(v4 + 1);
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
    return v3;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
}
