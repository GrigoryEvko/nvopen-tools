// Function: sub_18807D0
// Address: 0x18807d0
//
__int64 __fastcall sub_18807D0(__int64 a1, __int64 a2)
{
  unsigned __int64 *v3; // rax
  __int64 v4; // r8
  __int64 v5; // rdi
  unsigned __int64 v7; // rdx
  unsigned __int64 *v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rax
  __int64 v13; // rax
  unsigned __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_QWORD *)(a2 + 144) > *(_QWORD *)(a2 + 152) )
    *(_QWORD *)(a2 + 144) = 0;
  v3 = *(unsigned __int64 **)a2;
  v4 = a1 + 8;
  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v5 == *(_QWORD *)a2 )
  {
    *(_DWORD *)(a1 + 8) = 0;
    LOBYTE(_RCX) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = v4;
    *(_QWORD *)(a1 + 32) = v4;
    *(_QWORD *)(a1 + 40) = 0;
    v13 = *(_QWORD *)(a2 + 144);
    *(_DWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 48) = v13;
  }
  else
  {
    _RCX = 0;
    do
    {
      v7 = *v3++ - *(_QWORD *)(a2 + 144);
      *(v3 - 1) = v7;
      _RCX |= v7;
    }
    while ( (unsigned __int64 *)v5 != v3 );
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = v4;
    *(_QWORD *)(a1 + 32) = v4;
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 144);
    if ( _RCX )
    {
      __asm { tzcnt   rax, rcx }
      *(_DWORD *)(a1 + 64) = _RAX;
      LOBYTE(_RCX) = _RAX;
    }
    else
    {
      *(_DWORD *)(a1 + 64) = 0;
    }
  }
  *(_QWORD *)(a1 + 56) = ((*(_QWORD *)(a2 + 152) - *(_QWORD *)(a2 + 144)) >> _RCX) + 1LL;
  v9 = *(unsigned __int64 **)a2;
  v10 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v10 != *(_QWORD *)a2 )
  {
    while ( 1 )
    {
      v11 = *v9++;
      v14[0] = v11 >> _RCX;
      sub_A19EB0((_QWORD *)a1, v14);
      if ( (unsigned __int64 *)v10 == v9 )
        break;
      LODWORD(_RCX) = *(_DWORD *)(a1 + 64);
    }
  }
  return a1;
}
