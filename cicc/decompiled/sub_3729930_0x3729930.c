// Function: sub_3729930
// Address: 0x3729930
//
__int64 __fastcall sub_3729930(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax

  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(*(_QWORD *)(v3 + 208) + 336LL) == 3 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(v3 + 224) + 16LL) + 88LL))(*(_QWORD *)(*(_QWORD *)(v3 + 224) + 16LL));
    v3 = *(_QWORD *)(a1 + 8);
  }
  result = sub_31DB780(v3, a2);
  if ( (_DWORD)result == 2 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    if ( !*(_BYTE *)(a1 + 25) )
    {
      if ( *(_DWORD *)(v8 + 776) == 2 )
      {
        (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v8 + 224) + 856LL))(*(_QWORD *)(v8 + 224), 0, 1);
        v8 = *(_QWORD *)(a1 + 8);
      }
      *(_BYTE *)(a1 + 25) = 1;
    }
    *(_BYTE *)(a1 + 24) = 1;
    return sub_E9C600(*(__int64 **)(v8 + 224), 0, 0, v5, v6, v7);
  }
  return result;
}
