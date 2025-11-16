// Function: sub_164D6D0
// Address: 0x164d6d0
//
void __fastcall sub_164D6D0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // rdx
  const char *v5; // rax
  __int64 v6; // rdx
  const char *v7; // rbx
  const char *v8; // r8
  size_t v9; // rbx
  _BYTE *v10; // rdi
  int v11; // eax
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  const char *v14; // [rsp+0h] [rbp-150h]
  const char *v15; // [rsp+8h] [rbp-148h]
  _BYTE *v16; // [rsp+10h] [rbp-140h] BYREF
  __int64 v17; // [rsp+18h] [rbp-138h]
  _BYTE dest[304]; // [rsp+20h] [rbp-130h] BYREF

  v3 = (_QWORD *)sub_16498B0(a2);
  v4 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D19C0(a1, v3 + 2, *v3));
  if ( !*v4 )
    goto LABEL_11;
  if ( *v4 == -8 )
  {
    --*(_DWORD *)(a1 + 16);
LABEL_11:
    *v4 = v3;
    ++*(_DWORD *)(a1 + 12);
    sub_16D1CD0(a1, 0);
    return;
  }
  v5 = sub_1649960(a2);
  v7 = &v5[v6];
  v15 = &v5[v6];
  v16 = dest;
  v8 = sub_1649960(a2);
  v9 = v7 - v8;
  v17 = 0x10000000000LL;
  if ( v9 > 0x100 )
  {
    v14 = v8;
    sub_16CD150(&v16, dest, v9, 1);
    v11 = v17;
    v8 = v14;
    v10 = &v16[(unsigned int)v17];
  }
  else
  {
    v10 = dest;
    v11 = 0;
  }
  if ( v8 != v15 )
  {
    memcpy(v10, v8, v9);
    v11 = v17;
  }
  LODWORD(v17) = v11 + v9;
  v12 = sub_16498B0(a2);
  _libc_free(v12);
  v13 = sub_164D1F0(a1, a2, (__int64)&v16);
  sub_164B0D0(a2, v13);
  if ( v16 != dest )
    _libc_free((unsigned __int64)v16);
}
