// Function: sub_1681F50
// Address: 0x1681f50
//
__int64 __fastcall sub_1681F50(__int64 a1, const void *a2)
{
  _BYTE *v4; // rbx
  const char *v5; // rsi

  if ( !memcmp(a2, "__cuda_syscall", 0xEu) )
    return 1;
  if ( !a1 )
  {
    v4 = &off_4984DC8;
    v5 = "vprintf";
    while ( strcmp((const char *)a2, v5) )
    {
      if ( v4 == algn_4984ED0 )
        return 0;
      v5 = *(const char **)v4;
      v4 += 8;
    }
    return 1;
  }
  return sub_16849C0(*(_QWORD *)(a1 + 496), a2);
}
