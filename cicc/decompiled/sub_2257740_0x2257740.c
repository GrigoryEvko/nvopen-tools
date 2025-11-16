// Function: sub_2257740
// Address: 0x2257740
//
void __fastcall sub_2257740(__int64 a1, const char *a2)
{
  size_t v2; // rax
  size_t v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  volatile signed __int32 *v6; // rcx
  volatile signed __int32 *v7; // rdi
  int v8; // edx
  volatile signed __int32 *v9; // [rsp+8h] [rbp-30h] BYREF

  if ( !a2 )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v2 = strlen(a2);
  v3 = v2;
  if ( v2 )
  {
    v4 = sub_22153F0(v2, 0);
    v5 = v4;
    v6 = (volatile signed __int32 *)(v4 + 24);
    if ( v3 == 1 )
      *(_BYTE *)(v4 + 24) = *a2;
    else
      v6 = (volatile signed __int32 *)memcpy((void *)(v4 + 24), a2, v3);
    if ( (_UNKNOWN *)v5 != &unk_4FD67C0 )
    {
      *(_DWORD *)(v5 + 16) = 0;
      *(_QWORD *)v5 = v3;
      *(_BYTE *)(v5 + v3 + 24) = 0;
    }
  }
  else
  {
    v6 = (volatile signed __int32 *)&unk_4FD67D8;
  }
  v9 = v6;
  sub_2257710(a1, &v9);
  v7 = v9 - 6;
  if ( v9 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v8 = _InterlockedExchangeAdd(v9 - 2, 0xFFFFFFFF);
    }
    else
    {
      v8 = *((_DWORD *)v9 - 2);
      *((_DWORD *)v9 - 2) = v8 - 1;
    }
    if ( v8 <= 0 )
      j_j___libc_free_0_1((unsigned __int64)v7);
  }
}
