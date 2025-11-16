// Function: sub_10616A0
// Address: 0x10616a0
//
int __fastcall sub_10616A0(__int64 a1, const char *a2, unsigned __int64 a3)
{
  const char *v3; // rax
  __int64 v5; // rdx
  unsigned __int8 *v6; // rax
  unsigned __int8 *v7; // r14
  const char *v9; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v10; // [rsp-50h] [rbp-50h]
  __int16 v11; // [rsp-38h] [rbp-38h]

  LODWORD(v3) = (*(_BYTE *)(a1 + 32) & 0xF) - 7;
  if ( (unsigned int)v3 > 1 )
  {
    v3 = sub_BD5D20(a1);
    if ( a3 != v5 || a3 && (LODWORD(v3) = memcmp(v3, a2, a3), (_DWORD)v3) )
    {
      v6 = (unsigned __int8 *)sub_BA8B30(*(_QWORD *)(a1 + 40), (__int64)a2, a3);
      v7 = v6;
      if ( v6 )
      {
        sub_BD6B90((unsigned __int8 *)a1, v6);
        v9 = a2;
        v10 = a3;
        v11 = 261;
        LODWORD(v3) = sub_BD6B50(v7, &v9);
      }
      else
      {
        v9 = a2;
        v11 = 261;
        v10 = a3;
        LODWORD(v3) = sub_BD6B50((unsigned __int8 *)a1, &v9);
      }
    }
  }
  return (int)v3;
}
