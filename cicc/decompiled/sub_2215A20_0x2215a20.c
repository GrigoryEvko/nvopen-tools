// Function: sub_2215A20
// Address: 0x2215a20
//
void *__fastcall sub_2215A20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  size_t v4; // rdx
  __int64 v5; // rbp
  void *v6; // r8
  void *v7; // rax

  v3 = sub_22153F0(a3 + *(_QWORD *)a1, *(_QWORD *)(a1 + 8));
  v4 = *(_QWORD *)a1;
  v5 = v3;
  v6 = (void *)(v3 + 24);
  if ( *(_QWORD *)a1 )
  {
    if ( v4 == 1 )
    {
      *(_BYTE *)(v3 + 24) = *(_BYTE *)(a1 + 24);
      v4 = *(_QWORD *)a1;
      if ( (_UNKNOWN *)v3 == &unk_4FD67C0 )
        return v6;
      goto LABEL_7;
    }
    v7 = memcpy((void *)(v3 + 24), (const void *)(a1 + 24), v4);
    v4 = *(_QWORD *)a1;
    v6 = v7;
  }
  if ( (_UNKNOWN *)v5 == &unk_4FD67C0 )
    return v6;
LABEL_7:
  *(_DWORD *)(v5 + 16) = 0;
  *(_QWORD *)v5 = v4;
  *(_BYTE *)(v5 + v4 + 24) = 0;
  return v6;
}
