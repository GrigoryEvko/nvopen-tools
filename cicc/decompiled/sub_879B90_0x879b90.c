// Function: sub_879B90
// Address: 0x879b90
//
__int64 sub_879B90()
{
  __int64 v0; // rax
  __int64 result; // rax

  sub_666D40((__int64)"template<__edg_size_type__ N, typename ...T>  struct __type_pack_element;", 0);
  v0 = sub_879550("__type_pack_element", 0, 0x80000u);
  *(_BYTE *)(*(_QWORD *)(v0 + 88) + 266LL) |= 0x20u;
  unk_4D049C8 = v0;
  sub_666D40(
    (__int64)"template<__edg_size_type__ N, typename ...T>  __internal_alias_decl __type_pack_element_alias = int;",
    0);
  result = sub_879550("__type_pack_element_alias", 0, 0x80000u);
  *(_BYTE *)(*(_QWORD *)(result + 88) + 266LL) |= 0x20u;
  unk_4D049C0 = result;
  return result;
}
