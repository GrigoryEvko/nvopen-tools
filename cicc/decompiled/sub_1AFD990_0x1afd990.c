// Function: sub_1AFD990
// Address: 0x1afd990
//
__int64 __fastcall sub_1AFD990(__int64 a1, const void *a2, size_t a3)
{
  __int64 v3; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  _BYTE *v9; // rdi
  const void *v10; // rax
  __int64 v11; // rdx

  v3 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v3 <= 1 )
    return 0;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = 1;
  while ( 1 )
  {
    v8 = *(_QWORD *)(a1 + 8 * (v7 - v3));
    if ( (unsigned __int8)(*(_BYTE *)v8 - 4) <= 0x1Eu )
    {
      v9 = *(_BYTE **)(v8 - 8LL * *(unsigned int *)(v8 + 8));
      if ( !*v9 )
      {
        v10 = (const void *)sub_161E970((__int64)v9);
        if ( v11 == a3 && (!a3 || !memcmp(a2, v10, a3)) )
          break;
      }
    }
    if ( v6 == ++v7 )
      return 0;
    v3 = *(unsigned int *)(a1 + 8);
  }
  return v8;
}
