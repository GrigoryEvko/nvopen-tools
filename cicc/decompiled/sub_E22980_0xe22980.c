// Function: sub_E22980
// Address: 0xe22980
//
unsigned __int64 __fastcall sub_E22980(__int64 a1, size_t *a2)
{
  _QWORD *v3; // rax
  unsigned __int64 v4; // r12
  size_t v5; // r13
  _BYTE *v6; // rax
  size_t v7; // rbx
  size_t v8; // rsi
  size_t v9; // rbx
  size_t v10; // rcx
  __int64 *v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // [rsp+8h] [rbp-38h]

  sub_E20730(a2, 2u, "?A");
  v3 = *(_QWORD **)(a1 + 16);
  v4 = (*v3 + v3[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v3[1] = v4 - *v3 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v12 = (__int64 *)sub_22077B0(32);
    v13 = v12;
    if ( v12 )
    {
      *v12 = 0;
      v12[1] = 0;
      v12[2] = 0;
      v12[3] = 0;
    }
    v14 = sub_2207820(4096);
    v13[2] = 4096;
    *v13 = v14;
    v4 = v14;
    v15 = *(_QWORD *)(a1 + 16);
    v13[1] = 40;
    v13[3] = v15;
    *(_QWORD *)(a1 + 16) = v13;
  }
  if ( !v4 )
  {
    MEMORY[0x18] = 0;
    BUG();
  }
  *(_QWORD *)(v4 + 24) = 0;
  *(_QWORD *)(v4 + 32) = 0;
  *(_DWORD *)(v4 + 8) = 5;
  *(_QWORD *)v4 = &unk_49E0F88;
  *(_QWORD *)(v4 + 16) = 0;
  *(_QWORD *)(v4 + 24) = 21;
  *(_QWORD *)(v4 + 32) = "`anonymous namespace'";
  v5 = *a2;
  if ( *a2 && (v16 = (_BYTE *)a2[1], (v6 = memchr(v16, 64, *a2)) != 0) && (v7 = v6 - v16, v6 - v16 != -1) )
  {
    v8 = v5;
    if ( v5 > v7 )
      v8 = v6 - v16;
    v9 = v7 + 1;
    sub_E21AF0(a1, v8, v16);
    v10 = *a2;
    if ( v9 > *a2 )
      sub_222CF80("%s: __pos (which is %zu) > __size (which is %zu)", (char)"basic_string_view::substr");
    a2[1] += v9;
    *a2 = v10 - v9;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return v4;
}
