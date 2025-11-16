// Function: sub_EC6110
// Address: 0xec6110
//
__int64 __fastcall sub_EC6110(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // r13
  const char *v8; // rax
  __int64 v9; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  void *v13; // rax
  const char *v14; // [rsp+0h] [rbp-60h] BYREF
  const char *v15; // [rsp+8h] [rbp-58h]
  const char *v16[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v17; // [rsp+30h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v14 = 0;
  v15 = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v14);
  if ( (_BYTE)v3 )
  {
    v12 = *(_QWORD *)(a1 + 8);
    v16[0] = "expected identifier in directive";
    v17 = 259;
    return (unsigned int)sub_ECE0E0(v12, v16, 0, 0);
  }
  else
  {
    v4 = v3;
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v17 = 261;
    v16[0] = v14;
    v16[1] = v15;
    v6 = sub_E6C460(v5, v16);
    v7 = (_QWORD *)v6;
    if ( *(_QWORD *)v6
      || (*(_BYTE *)(v6 + 9) & 0x70) == 0x20
      && *(char *)(v6 + 8) >= 0
      && (*(_BYTE *)(v6 + 8) |= 8u, v13 = sub_E807D0(*(_QWORD *)(v6 + 24)), (*v7 = v13) != 0) )
    {
      HIBYTE(v17) = 1;
      v8 = ".alt_entry must preceed symbol definition";
LABEL_4:
      v9 = *(_QWORD *)(a1 + 8);
      v16[0] = v8;
      LOBYTE(v17) = 3;
      return (unsigned int)sub_ECE0E0(v9, v16, 0, 0);
    }
    v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v11 + 296LL))(v11, v7, 20) )
    {
      HIBYTE(v17) = 1;
      v8 = "unable to emit symbol attribute";
      goto LABEL_4;
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    return v4;
  }
}
