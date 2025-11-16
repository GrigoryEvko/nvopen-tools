// Function: sub_393F820
// Address: 0x393f820
//
__int64 __fastcall sub_393F820(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  char *v8; // rax
  int v9; // eax
  __int64 v10; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax

  if ( sub_393F120(*a2) )
  {
    v5 = *a2;
    *a2 = 0;
    v6 = (_QWORD *)sub_22077B0(0x70u);
    v7 = v6;
    if ( v6 )
    {
      *v6 = &unk_4A3F020;
      sub_16D1950((__int64)(v6 + 1), 0, 136);
      v7[5] = a3;
      v8 = (char *)&unk_4A3F070;
      v7[6] = v5;
      v7[7] = 0;
      *((_DWORD *)v7 + 16) = 255;
      v7[9] = 0;
      v7[10] = 0;
      goto LABEL_4;
    }
  }
  else if ( sub_393F190(*a2) )
  {
    v5 = *a2;
    *a2 = 0;
    v12 = (_QWORD *)sub_22077B0(0x70u);
    v7 = v12;
    if ( v12 )
    {
      *v12 = &unk_4A3F020;
      sub_16D1950((__int64)(v12 + 1), 0, 136);
      v7[5] = a3;
      v8 = (char *)&unk_4A3F0B8;
      v7[6] = v5;
      v7[7] = 0;
      *((_DWORD *)v7 + 16) = 2;
      v7[9] = 0;
      v7[10] = 0;
LABEL_4:
      v7[11] = 0;
      *v7 = v8 + 16;
      v7[12] = 0;
      v7[13] = 0;
      v9 = sub_3940DC0(v7);
      if ( v9 )
        goto LABEL_5;
      goto LABEL_12;
    }
  }
  else if ( (unsigned __int8)sub_393F7E0(*a2) )
  {
    v5 = *a2;
    *a2 = 0;
    v19 = (_QWORD *)sub_22077B0(0x70u);
    v7 = v19;
    if ( v19 )
    {
      *v19 = &unk_4A3F020;
      sub_16D1950((__int64)(v19 + 1), 0, 136);
      v7[5] = a3;
      v7[6] = v5;
      v7[9] = v5;
      *v7 = &unk_4A3F110;
      v7[7] = 0;
      *((_DWORD *)v7 + 16) = 3;
      v7[10] = 0;
      v7[11] = 0;
      v7[12] = 0;
      v7[13] = 0;
      v9 = ((__int64 (__fastcall *)(_QWORD *))sub_393F300)(v7);
      goto LABEL_11;
    }
  }
  else
  {
    v13 = *a2;
    if ( !(unsigned __int8)sub_393EF60(*a2) )
    {
      *(_BYTE *)(a1 + 16) |= 1u;
      v20 = sub_393D180(v13, (__int64)a2, v14, v15, v16, v17);
      *(_DWORD *)a1 = 6;
      *(_QWORD *)(a1 + 8) = v20;
      return a1;
    }
    v5 = *a2;
    *a2 = 0;
    v18 = (_QWORD *)sub_22077B0(0x48u);
    v7 = v18;
    if ( v18 )
    {
      *v18 = &unk_4A3F020;
      sub_16D1950((__int64)(v18 + 1), 0, 136);
      v7[5] = a3;
      v7[6] = v5;
      v7[7] = 0;
      *v7 = &unk_4A3F050;
      *((_DWORD *)v7 + 16) = 1;
      v9 = ((__int64 (__fastcall *)(_QWORD *))sub_393D700)(v7);
      goto LABEL_11;
    }
  }
  if ( v5 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  v9 = (*(__int64 (__fastcall **)(_QWORD *))(MEMORY[0] + 16LL))(v7);
LABEL_11:
  if ( v9 )
  {
LABEL_5:
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v9;
    *(_QWORD *)(a1 + 8) = v10;
    if ( v7 )
      (*(void (__fastcall **)(_QWORD *))(*v7 + 8LL))(v7);
    return a1;
  }
LABEL_12:
  *(_QWORD *)a1 = v7;
  *(_BYTE *)(a1 + 16) &= ~1u;
  return a1;
}
