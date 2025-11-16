// Function: sub_E96F00
// Address: 0xe96f00
//
unsigned __int64 __fastcall sub_E96F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned __int64 result; // rax
  void *v9; // rdx
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  unsigned __int64 v14; // rax
  _BYTE *v15; // rdi
  unsigned __int8 *v16; // rsi
  size_t v17; // r12
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD v21[2]; // [rsp+0h] [rbp-30h] BYREF
  int v22; // [rsp+10h] [rbp-20h]

  v6 = a4;
  result = *(unsigned __int8 *)(a1 + 188);
  if ( (unsigned __int8)(result - 2) <= 1u )
  {
    if ( *(_BYTE *)(a1 + 148) )
      sub_C64ED0("Unhandled storage-mapping class for .text csect", 1u);
    return (unsigned __int64)sub_E96DC0(a1, a4);
  }
  if ( (unsigned __int8)(result - 4) <= 7u )
  {
    v18 = *(_BYTE *)(a1 + 148);
    if ( v18 != 16 && v18 != 1 )
      sub_C64ED0("Unhandled storage-mapping class for .rodata csect.", 1u);
    return (unsigned __int64)sub_E96DC0(a1, a4);
  }
  switch ( (_BYTE)result )
  {
    case 0x14:
      if ( (*(_BYTE *)(a1 + 148) & 0xFB) != 1 && *(_BYTE *)(a1 + 148) != 16 )
        sub_C64ED0("Unexepected storage-mapping class for ReadOnlyWithRel kind", 1u);
      return (unsigned __int64)sub_E96DC0(a1, a4);
    case 0xD:
      if ( *(_BYTE *)(a1 + 148) != 20 )
        sub_C64ED0("Unhandled storage-mapping class for .tdata csect.", 1u);
      return (unsigned __int64)sub_E96DC0(a1, a4);
    case 0x13:
      result = (unsigned __int8)(*(_BYTE *)(a1 + 148) - 3);
      switch ( *(_BYTE *)(a1 + 148) )
      {
        case 3:
        case 0x16:
          return result;
        case 5:
        case 0xA:
        case 0x10:
          return (unsigned __int64)sub_E96DC0(a1, a4);
        case 0xF:
          v19 = *(_QWORD *)(a4 + 32);
          result = *(_QWORD *)(a4 + 24) - v19;
          if ( result <= 5 )
            return sub_CB6200(a4, "\t.toc\n", 6u);
          *(_DWORD *)v19 = 1869884937;
          *(_WORD *)(v19 + 4) = 2659;
          *(_QWORD *)(a4 + 32) += 6LL;
          return result;
        default:
          sub_C64ED0("Unhandled storage-mapping class for .data csect.", 1u);
      }
  }
  if ( *(_BYTE *)(a1 + 150) )
  {
    if ( *(_BYTE *)(a1 + 148) == 16 )
    {
      if ( (_BYTE)result == 18 )
        return result;
      return (unsigned __int64)sub_E96DC0(a1, a4);
    }
    if ( *(_BYTE *)(a1 + 149) == 3 )
      return result;
  }
  if ( (result & 0xFD) == 0xC )
    return (unsigned __int64)sub_E96DC0(a1, a4);
  if ( (_BYTE)result || !*(_BYTE *)(a1 + 180) )
    sub_C64ED0("Printing for this SectionKind is unimplemented.", 1u);
  v9 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 <= 9u )
  {
    v10 = sub_CB6200(a4, "\n\t.dwsect ", 0xAu);
  }
  else
  {
    v10 = a4;
    qmemcpy(v9, "\n\t.dwsect ", 10);
    *(_QWORD *)(a4 + 32) += 10LL;
  }
  v11 = *(_DWORD *)(a1 + 176);
  v21[1] = "0x%x";
  v22 = v11;
  v21[0] = &unk_49E3678;
  v12 = sub_CB6620(v10, (__int64)v21, (__int64)&unk_49E3678, a4, a5, a6);
  v13 = *(_BYTE **)(v12 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
  {
    sub_CB5D20(v12, 10);
  }
  else
  {
    *(_QWORD *)(v12 + 32) = v13 + 1;
    *v13 = 10;
  }
  v14 = *(_QWORD *)(v6 + 24);
  v15 = *(_BYTE **)(v6 + 32);
  v16 = *(unsigned __int8 **)(a1 + 128);
  v17 = *(_QWORD *)(a1 + 136);
  if ( v17 > v14 - (unsigned __int64)v15 )
  {
    v20 = sub_CB6200(v6, v16, v17);
    v15 = *(_BYTE **)(v20 + 32);
    v6 = v20;
    v14 = *(_QWORD *)(v20 + 24);
  }
  else if ( v17 )
  {
    memcpy(v15, v16, v17);
    v14 = *(_QWORD *)(v6 + 24);
    v15 = (_BYTE *)(v17 + *(_QWORD *)(v6 + 32));
    *(_QWORD *)(v6 + 32) = v15;
  }
  if ( v14 <= (unsigned __int64)v15 )
  {
    v6 = sub_CB5D20(v6, 58);
  }
  else
  {
    *(_QWORD *)(v6 + 32) = v15 + 1;
    *v15 = 58;
  }
  result = *(_QWORD *)(v6 + 32);
  if ( result >= *(_QWORD *)(v6 + 24) )
    return sub_CB5D20(v6, 10);
  *(_QWORD *)(v6 + 32) = result + 1;
  *(_BYTE *)result = 10;
  return result;
}
