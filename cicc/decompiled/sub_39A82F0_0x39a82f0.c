// Function: sub_39A82F0
// Address: 0x39a82f0
//
__int64 __fastcall sub_39A82F0(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  __int64 v4; // r12
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  const char *v10; // r15
  void *v11; // rcx
  size_t v12; // rdx
  size_t v13; // r8

  v3 = sub_39A81B0((__int64)a1, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))));
  v4 = (__int64)sub_39A23D0((__int64)a1, (unsigned __int8 *)a2);
  if ( !v4 )
  {
    v4 = sub_39A5A90((__int64)a1, 57, (__int64)v3, (unsigned __int8 *)a2);
    v6 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
    if ( v6 && (v7 = sub_161E970(v6), v9 = v8, v10 = (const char *)v7, v8) )
    {
      v11 = *(void **)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
      if ( v11 )
      {
        v11 = (void *)sub_161E970((__int64)v11);
        v13 = v12;
      }
      else
      {
        v13 = 0;
      }
      sub_39A3F30(a1, v4, 3, v11, v13);
    }
    else
    {
      v10 = "(anonymous namespace)";
      v9 = 21;
    }
    sub_3990460(a1[25], (__int64)v10, v9);
    (*(void (__fastcall **)(__int64 *, const char *, __int64, __int64, _QWORD))(*a1 + 8))(
      a1,
      v10,
      v9,
      v4,
      *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))));
    if ( (*(_BYTE *)(a2 + 24) & 1) != 0 )
      sub_39A34D0((__int64)a1, v4, 137);
  }
  return v4;
}
