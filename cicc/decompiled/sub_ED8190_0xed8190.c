// Function: sub_ED8190
// Address: 0xed8190
//
_QWORD *__fastcall sub_ED8190(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F8A428) )
  {
    v4 = *a2;
    *a2 = 0;
    if ( *(_DWORD *)(v4 + 8) == 13 )
    {
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)a3 + 136LL) + 24LL))(
        v10,
        *(_QWORD *)(*(_QWORD *)a3 + 136LL),
        **(_QWORD **)(a3 + 8),
        *(_QWORD *)(*(_QWORD *)(a3 + 8) + 8LL),
        *(_QWORD *)(a3 + 16));
      v5 = v10[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v10[0] = 0;
        *a1 = v5 | 1;
      }
      else
      {
        *a1 = 1;
        v10[0] = 0;
      }
      sub_9C66B0(v10);
    }
    else
    {
      v8 = sub_22077B0(48);
      v9 = v8;
      if ( v8 )
      {
        *(_QWORD *)v8 = &unk_49E4BC8;
        *(_DWORD *)(v8 + 8) = *(_DWORD *)(v4 + 8);
        *(_QWORD *)(v8 + 16) = v8 + 32;
        sub_ED71E0((__int64 *)(v8 + 16), *(_BYTE **)(v4 + 16), *(_QWORD *)(v4 + 16) + *(_QWORD *)(v4 + 24));
      }
      *a1 = v9 | 1;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    v6 = *a2;
    *a2 = 0;
    *a1 = v6 | 1;
  }
  return a1;
}
