// Function: sub_20C8B80
// Address: 0x20c8b80
//
__int64 __fastcall sub_20C8B80(__int64 a1, _BYTE *a2)
{
  unsigned __int64 v2; // r12
  __int64 v3; // r15
  unsigned __int64 v4; // r13
  char v5; // al
  _QWORD *i; // rbx
  __int64 v8; // rax
  __int64 (*v9)(); // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  const char **v13; // rcx
  __int64 v14; // rdi
  __int64 (*v15)(); // rax

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40);
  v4 = sub_157EBA0(v3);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 25 )
  {
    if ( (a2[793] & 1) == 0 || v5 != 31 )
      return 0;
    v4 = 0;
  }
  if ( (unsigned __int8)sub_15F3040(v2)
    || sub_15F3330(v2)
    || (unsigned __int8)sub_15F2ED0(v2)
    || !(unsigned __int8)sub_14AF470(v2, 0, 0, 0) )
  {
    for ( i = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 40) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
          ;
          i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( !i )
        BUG();
      if ( (_QWORD *)v2 == i - 3 )
        break;
      if ( *((_BYTE *)i - 8) == 78 )
      {
        v8 = *(i - 6);
        if ( !*(_BYTE *)(v8 + 16) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v8 + 36) - 35) <= 3 )
          continue;
      }
      if ( (unsigned __int8)sub_15F3040((__int64)(i - 3))
        || sub_15F3330((__int64)(i - 3))
        || (unsigned __int8)sub_15F2ED0((__int64)(i - 3))
        || !(unsigned __int8)sub_14AF470((__int64)(i - 3), 0, 0, 0) )
      {
        return 0;
      }
    }
  }
  v9 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
  if ( v9 == sub_16FF750 )
    BUG();
  v10 = *(_QWORD *)(v3 + 56);
  v11 = ((__int64 (__fastcall *)(_BYTE *, __int64))v9)(a2, v10);
  v13 = 0;
  v14 = v11;
  v15 = *(__int64 (**)())(*(_QWORD *)v11 + 56LL);
  if ( v15 != sub_1D12D20 )
    v13 = (const char **)((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v15)(v14, v10, v12, 0);
  return sub_20C8B40(v10, v2, v4, v13);
}
