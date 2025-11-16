// Function: sub_8129A0
// Address: 0x8129a0
//
__int64 __fastcall sub_8129A0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r13
  __int64 result; // rax

  v3 = a1;
  if ( a2 && !*(_BYTE *)(a2 + 32) && *(_QWORD *)(a2 + 16) )
  {
    v3 = *(_QWORD *)(a2 + 16);
    goto LABEL_5;
  }
  if ( !(unsigned int)sub_8D2EF0(a1) )
  {
LABEL_5:
    if ( dword_4D0425C )
      goto LABEL_6;
LABEL_13:
    sub_8128F0(a2, a3);
    *a3 += 2LL;
    sub_8238B0(qword_4F18BE0, "dn", 2);
    return sub_80F5E0(v3, 0, a3);
  }
  v3 = sub_8D46C0(a1);
  if ( !dword_4D0425C )
    goto LABEL_13;
LABEL_6:
  if ( (unsigned int)sub_8DBE70(v3) )
  {
    if ( a2 )
    {
      if ( *(_QWORD *)(a2 + 8) )
      {
        *a3 += 2LL;
        sub_8238B0(qword_4F18BE0, "sr", 2);
        sub_80F5E0(v3, 0, a3);
      }
    }
    *a3 += 2LL;
    sub_8238B0(qword_4F18BE0, "co", 2);
    ++a3[5];
    result = sub_80F5E0(v3, 0, a3);
    --a3[5];
  }
  else
  {
    *a3 += 4LL;
    sub_8238B0(qword_4F18BE0, "L_ZN", 4);
    sub_80F5E0(v3, 0, a3);
    *a3 += 2LL;
    sub_8238B0(qword_4F18BE0, "D1", 2);
    *a3 += 3LL;
    return sub_8238B0(qword_4F18BE0, "EvE", 3);
  }
  return result;
}
