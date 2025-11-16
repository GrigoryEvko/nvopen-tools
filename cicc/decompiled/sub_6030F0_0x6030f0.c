// Function: sub_6030F0
// Address: 0x6030f0
//
void __fastcall sub_6030F0(unsigned int *a1)
{
  __int64 v2; // rdi
  char **v3; // r13
  char **v4; // rax
  _BYTE v5[4]; // [rsp+0h] [rbp-30h] BYREF
  _BYTE v6[4]; // [rsp+4h] [rbp-2Ch] BYREF
  _BYTE v7[4]; // [rsp+8h] [rbp-28h] BYREF
  _BYTE v8[36]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = *a1;
  if ( (_DWORD)v2 != dword_4F063F8 && unk_4F04C48 == -1 )
  {
    v3 = (char **)sub_729B10(v2, v5, v7, 0);
    v4 = (char **)sub_729B10(dword_4F063F8, v6, v8, 0);
    if ( v3 != 0 && v3 != v4 && v4 && *v3 && *v4 )
    {
      if ( (unsigned int)sub_722E50(*v3, *v4) )
        sub_684B30(1644, a1);
    }
  }
}
