// Function: sub_1695830
// Address: 0x1695830
//
int __fastcall sub_1695830(__int64 a1, void *a2, size_t a3)
{
  const void *v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // r15
  __int64 v7; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_1649960(a1);
  if ( v5 != a3 || a3 && (LODWORD(v4) = memcmp(a2, v4, a3), (_DWORD)v4) )
  {
    v4 = (const void *)sub_1695640(a1);
    if ( !v4 )
    {
      v6 = (__int64 *)sub_15E0530(a1);
      v9[0] = sub_161FF10(v6, a2, a3);
      v7 = sub_1627350(v6, v9, (__int64 *)1, 0, 1);
      LODWORD(v4) = sub_1627100(a1, "PGOFuncName", 0xBu, v7);
    }
  }
  return (int)v4;
}
