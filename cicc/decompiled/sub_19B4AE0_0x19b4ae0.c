// Function: sub_19B4AE0
// Address: 0x19b4ae0
//
__int64 __fastcall sub_19B4AE0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rdx
  _BYTE *v10; // rdi
  const void *v11; // rax
  size_t v12; // rdx

  v4 = sub_13FD000(a1);
  if ( !v4 )
    return 0;
  v5 = v4;
  v6 = *(unsigned int *)(v4 + 8);
  if ( (unsigned int)v6 <= 1 )
    return 0;
  v7 = v6;
  v8 = 1;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v5 + 8 * (v8 - v6));
    if ( (unsigned __int8)(*(_BYTE *)v9 - 4) <= 0x1Eu )
    {
      v10 = *(_BYTE **)(v9 - 8LL * *(unsigned int *)(v9 + 8));
      if ( !*v10 )
      {
        v11 = (const void *)sub_161E970((__int64)v10);
        if ( a3 <= v12 && (!a3 || !memcmp(v11, a2, a3)) )
          break;
      }
    }
    if ( v7 == ++v8 )
      return 0;
    v6 = *(unsigned int *)(v5 + 8);
  }
  return 1;
}
