// Function: sub_1E869F0
// Address: 0x1e869f0
//
_BYTE *__fastcall sub_1E869F0(__int64 a1, const char *a2, __int64 a3)
{
  void *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rdx
  _BYTE *v8; // rax
  char *v9; // rax
  size_t v10; // rdx
  void *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  void *v15; // rax
  __int64 v16; // r12
  _BYTE *v17; // rax
  _QWORD *v18; // rdi
  _BYTE *result; // rax
  size_t v20; // [rsp+8h] [rbp-48h]
  _QWORD v21[2]; // [rsp+10h] [rbp-40h] BYREF
  void (__fastcall *v22)(_QWORD *, _QWORD *, __int64); // [rsp+20h] [rbp-30h]

  sub_1E857B0(a1, a2, *(__int64 **)(a3 + 56));
  v5 = sub_16E8CB0();
  v6 = sub_1263B40((__int64)v5, "- basic block: ");
  sub_1DD5B60(v21, a3);
  sub_1E869D0((__int64)v21, v6, v7);
  v8 = *(_BYTE **)(v6 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v6 + 16) )
  {
    v6 = sub_16E7DE0(v6, 32);
  }
  else
  {
    *(_QWORD *)(v6 + 24) = v8 + 1;
    *v8 = 32;
  }
  v9 = (char *)sub_1DD6290(a3);
  v11 = *(void **)(v6 + 24);
  if ( v10 > *(_QWORD *)(v6 + 16) - (_QWORD)v11 )
  {
    v6 = sub_16E7EE0(v6, v9, v10);
  }
  else if ( v10 )
  {
    v20 = v10;
    memcpy(v11, v9, v10);
    *(_QWORD *)(v6 + 24) += v20;
  }
  v12 = sub_1263B40(v6, " (");
  v13 = sub_16E7B40(v12, a3);
  v14 = *(_BYTE **)(v13 + 24);
  if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
  {
    sub_16E7DE0(v13, 41);
  }
  else
  {
    *(_QWORD *)(v13 + 24) = v14 + 1;
    *v14 = 41;
  }
  if ( v22 )
    v22(v21, v21, 3);
  if ( *(_QWORD *)(a1 + 584) )
  {
    v15 = sub_16E8CB0();
    v16 = sub_1263B40((__int64)v15, " [");
    v21[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 584) + 392LL) + 16LL * *(unsigned int *)(a3 + 48));
    sub_1F10810(v21, v16);
    v17 = *(_BYTE **)(v16 + 24);
    if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 16) )
    {
      v16 = sub_16E7DE0(v16, 59);
    }
    else
    {
      *(_QWORD *)(v16 + 24) = v17 + 1;
      *v17 = 59;
    }
    v21[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 584) + 392LL) + 16LL * *(unsigned int *)(a3 + 48) + 8);
    sub_1F10810(v21, v16);
    sub_1549FC0(v16, 0x29u);
  }
  v18 = sub_16E8CB0();
  result = (_BYTE *)v18[3];
  if ( (unsigned __int64)result >= v18[2] )
    return (_BYTE *)sub_16E7DE0((__int64)v18, 10);
  v18[3] = result + 1;
  *result = 10;
  return result;
}
