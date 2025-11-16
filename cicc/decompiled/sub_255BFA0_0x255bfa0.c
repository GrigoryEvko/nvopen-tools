// Function: sub_255BFA0
// Address: 0x255bfa0
//
__int64 __fastcall sub_255BFA0(__int64 a1, _BYTE *a2)
{
  __int64 (*v2)(void); // rax
  char v3; // r14
  __int64 (__fastcall *v4)(__int64); // rax
  char v5; // al
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // al
  __int64 (__fastcall *v8)(__int64); // rax
  char v9; // al
  __int64 v10; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( (char *)v2 == (char *)sub_2505E60 )
    v3 = *(_BYTE *)(a1 + 17);
  else
    v3 = v2();
  v4 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL);
  if ( v4 == sub_2505E60 )
    v5 = a2[17];
  else
    v5 = v4((__int64)a2);
  if ( v3 != v5 )
    return 0;
  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL);
  if ( v6 == sub_2505E60 )
    v7 = *(_BYTE *)(a1 + 17);
  else
    v7 = v6(a1);
  if ( v7
    || ((v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL), v8 != sub_2505E60)
      ? (v9 = v8((__int64)a2))
      : (v9 = a2[17]),
        v9) )
  {
    if ( *(_BYTE *)(a1 + 264) != a2[264] )
      return 0;
    v10 = *(unsigned int *)(a1 + 64);
    if ( v10 != *((_DWORD *)a2 + 16) )
      return 0;
    v12 = *(_QWORD *)(a1 + 56);
    v13 = *((_QWORD *)a2 + 7);
    v14 = v12 + 24 * v10;
    if ( v12 != v14 )
    {
      while ( *(_QWORD *)v12 == *(_QWORD *)v13
           && *(_QWORD *)(v12 + 8) == *(_QWORD *)(v13 + 8)
           && *(_BYTE *)(v12 + 16) == *(_BYTE *)(v13 + 16) )
      {
        v12 += 24;
        v13 += 24;
        if ( v14 == v12 )
          return 1;
      }
      return 0;
    }
  }
  return 1;
}
