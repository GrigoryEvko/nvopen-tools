// Function: sub_234D100
// Address: 0x234d100
//
__int64 __fastcall sub_234D100(
        __int64 a1,
        void (__fastcall *a2)(__int64, const void *, __int64),
        const void *a3,
        __int64 a4,
        const void *a5,
        size_t a6)
{
  const void *v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]

  v7 = a3;
  v8 = a4;
  if ( !(unsigned __int8)sub_95CB50(&v7, a5, a6)
    || v8 && (!(unsigned __int8)sub_95CB50(&v7, "<", 1u) || !(unsigned __int8)sub_232E070(&v7, ">", 1u)) )
  {
    BUG();
  }
  a2(a1, v7, v8);
  return a1;
}
