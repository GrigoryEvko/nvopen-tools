// Function: sub_AF47B0
// Address: 0xaf47b0
//
__int64 __fastcall sub_AF47B0(__int64 a1, unsigned __int64 *a2, unsigned __int64 *a3)
{
  unsigned __int64 *v3; // rbx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 *v8; // [rsp+8h] [rbp-28h] BYREF

  v8 = a2;
  if ( a3 == a2 )
  {
LABEL_6:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    v3 = a2;
    while ( *v3 != 4096 )
    {
      v3 += (unsigned int)sub_AF4160(&v8);
      v8 = v3;
      if ( a3 == v3 )
        goto LABEL_6;
    }
    v5 = v3[1];
    v6 = v3[2];
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)a1 = v6;
    return a1;
  }
}
