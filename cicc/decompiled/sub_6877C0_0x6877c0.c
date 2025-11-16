// Function: sub_6877C0
// Address: 0x6877c0
//
__int64 sub_6877C0()
{
  __int64 v0; // r8
  __int64 v1; // rdx
  __int64 v2; // rcx
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rax

  v0 = unk_4F04C50;
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( !unk_4F04C50 )
  {
    while ( 1 )
    {
      v2 = v1;
      if ( (*(_BYTE *)(v1 + 5) & 8) == 0 )
        break;
      v3 = *(_BYTE *)(v1 + 4);
      if ( v3 == 1 )
      {
        v3 = *(_BYTE *)(v1 - 772);
        v2 = v1 - 776;
      }
      if ( (unsigned __int8)(v3 - 6) > 1u
        || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v2 + 208) + 168LL) + 109LL) & 0x20) == 0 )
      {
        break;
      }
      v4 = *(int *)(v2 - 376);
      v1 = v2 - 776;
      if ( (_DWORD)v4 != -1 )
      {
        v5 = *(_QWORD *)(qword_4F04C68[0] + 776 * v4 + 184);
        if ( v5 )
          return v5;
      }
    }
  }
  return v0;
}
