// Function: sub_822B10
// Address: 0x822b10
//
__int64 __fastcall sub_822B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx

  v6 = malloc(a1, a2, a3, a4, a5, a6);
  if ( !v6 )
    goto LABEL_6;
  v11 = v6;
  v12 = dword_4F19600;
  if ( dword_4F19600 > 1023 )
  {
    v13 = (__int64 *)malloc(24, a2, v7, v8, v9, v10);
    if ( v13 )
      goto LABEL_4;
LABEL_6:
    sub_685240(4u);
  }
  ++dword_4F19600;
  v13 = (__int64 *)((char *)&unk_4F19620 + 24 * v12);
LABEL_4:
  v14 = qword_4F195F8;
  v13[1] = v11;
  v13[2] = a1;
  *v13 = v14;
  qword_4F195F8 = (__int64)v13;
  return v11;
}
