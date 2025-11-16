// Function: sub_12A4F30
// Address: 0x12a4f30
//
__int64 __fastcall sub_12A4F30(_QWORD *a1, _DWORD *a2)
{
  __int64 v3; // r13
  __int64 v4; // rdi
  _QWORD *v5; // rsi
  __int64 result; // rax
  __int64 v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rdx
  __int64 v13; // rdx

  v3 = a1[7];
  if ( v3 )
  {
    if ( sub_157EBA0(a1[7]) )
      sub_127B550("unexpected: last basic block has terminator!", a2, 1);
    v4 = a1[16];
    if ( v3 + 40 != (*(_QWORD *)(v3 + 40) & 0xFFFFFFFFFFFFFFF8LL) && *(_QWORD *)(v4 + 8) )
    {
      v5 = (_QWORD *)a1[16];
      return sub_1290AF0(a1, v5, 0);
    }
    result = sub_164D160(v4, v3);
    v7 = a1[16];
    if ( v7 )
    {
LABEL_9:
      sub_157EF40(v7);
      return j_j___libc_free_0(v7, 64);
    }
  }
  else
  {
    v8 = (_QWORD *)a1[16];
    v9 = v8[1];
    if ( !v9
      || *(_QWORD *)(v9 + 8)
      || (v10 = sub_1648700(v9), *(_BYTE *)(v10 + 16) != 26)
      || (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 1
      || (v12 = *(_QWORD **)(v10 - 24), v8 != v12)
      || !v12 )
    {
      v5 = v8;
      return sub_1290AF0(a1, v5, 0);
    }
    v13 = *(_QWORD *)(v10 + 40);
    a1[7] = v13;
    v13 += 40;
    a1[8] = v13;
    result = sub_15F20C0(v10, a2, v13, v11);
    v7 = a1[16];
    if ( v7 )
      goto LABEL_9;
  }
  return result;
}
