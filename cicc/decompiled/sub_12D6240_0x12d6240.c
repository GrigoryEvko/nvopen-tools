// Function: sub_12D6240
// Address: 0x12d6240
//
__int64 __fastcall sub_12D6240(__int64 a1, unsigned int a2, const char *a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  char v6; // r12
  bool v7; // al
  __int64 v8; // rdx
  const char *v10; // [rsp+0h] [rbp-40h] BYREF
  size_t v11; // [rsp+8h] [rbp-38h]
  __int64 *v12; // [rsp+10h] [rbp-30h] BYREF
  __int64 v13; // [rsp+20h] [rbp-20h] BYREF

  v4 = sub_12D6170(a1 + 120, a2);
  v5 = v4;
  if ( v4 )
  {
    if ( !*(_DWORD *)(v4 + 56) )
    {
      v7 = 1;
LABEL_10:
      v8 = *(unsigned int *)(v5 + 40);
      return (v8 << 32) | v7;
    }
    a3 = **(const char ***)(v4 + 48);
  }
  v10 = a3;
  if ( a3 && (v11 = strlen(a3)) != 0 )
  {
    sub_16D2060(&v12, &v10);
    v6 = *(_BYTE *)v12;
    if ( v12 != &v13 )
      j_j___libc_free_0(v12, v13 + 1);
    v7 = v6 == 49 || v6 == 116;
  }
  else
  {
    v7 = 1;
  }
  v8 = 0;
  if ( v5 )
    goto LABEL_10;
  return (v8 << 32) | v7;
}
