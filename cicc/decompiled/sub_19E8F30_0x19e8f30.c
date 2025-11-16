// Function: sub_19E8F30
// Address: 0x19e8f30
//
__int64 __fastcall sub_19E8F30(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r13d
  int v7; // r13d
  unsigned int v8; // r14d
  int v9; // eax
  __int64 v10; // rdi
  __int64 *v11; // r8
  int v12; // r9d
  unsigned int i; // edx
  __int64 *v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 v17; // [rsp+8h] [rbp-B8h]
  __int64 v18; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v19; // [rsp+88h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = v4 - 1;
  v17 = *(_QWORD *)(a1 + 8);
  v8 = ((unsigned int)a2[1] >> 9) ^ ((unsigned int)a2[1] >> 4);
  LODWORD(v18) = ((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4);
  v19 = sub_15AF870();
  HIDWORD(v18) = v8;
  v9 = sub_1593600(&v18, 8u, v19);
  v10 = *a2;
  v11 = 0;
  v12 = 1;
  for ( i = v7 & v9; ; i = v7 & v16 )
  {
    v14 = (__int64 *)(v17 + 16LL * i);
    v15 = *v14;
    if ( v10 == *v14 && a2[1] == v14[1] )
    {
      *a3 = v14;
      return 1;
    }
    if ( v15 == -8 )
      break;
    if ( v15 == -16 && v14[1] == -16 && !v11 )
      v11 = (__int64 *)(v17 + 16LL * i);
LABEL_10:
    v16 = v12 + i;
    ++v12;
  }
  if ( v14[1] != -8 )
    goto LABEL_10;
  if ( !v11 )
    v11 = (__int64 *)(v17 + 16LL * i);
  *a3 = v11;
  return 0;
}
