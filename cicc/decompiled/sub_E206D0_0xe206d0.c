// Function: sub_E206D0
// Address: 0xe206d0
//
__int64 __fastcall sub_E206D0(size_t a1, const void *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  size_t *v7; // rax
  unsigned int v8; // r12d
  size_t v9; // rdx
  _QWORD v11[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v12[3]; // [rsp+10h] [rbp-20h] BYREF

  v7 = v12;
  v12[0] = a3;
  if ( !a7 )
    v7 = v11;
  v12[1] = a4;
  v8 = 0;
  v11[0] = a5;
  v11[1] = a6;
  v9 = *v7;
  if ( *v7 <= a1 && (!v9 || !memcmp(a2, (const void *)v7[1], v9)) )
    return 1;
  return v8;
}
