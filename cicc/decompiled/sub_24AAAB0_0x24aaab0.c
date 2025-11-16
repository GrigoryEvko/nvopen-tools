// Function: sub_24AAAB0
// Address: 0x24aaab0
//
__int64 *__fastcall sub_24AAAB0(__int64 *a1, __int64 **a2, __int64 **a3)
{
  __int64 *v4; // r14
  __int64 *v6; // rax

  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, void *))(**a2 + 48))(*a2, &unk_4F8A428) )
  {
    v4 = *a2;
    *a2 = 0;
    sub_24A9E20(a3, v4);
    *a1 = 1;
    if ( v4 )
      (*(void (__fastcall **)(__int64 *))(*v4 + 8))(v4);
    return a1;
  }
  else
  {
    v6 = *a2;
    *a2 = 0;
    *a1 = (unsigned __int64)v6 | 1;
    return a1;
  }
}
