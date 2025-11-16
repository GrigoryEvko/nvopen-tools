// Function: sub_1369ED0
// Address: 0x1369ed0
//
__int64 __fastcall sub_1369ED0(__int64 a1, __int64 a2, unsigned int *a3)
{
  unsigned __int64 v3; // rbx
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // r12d
  int v13; // edx
  __int64 v14; // rax
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // esi
  __int64 *v19; // rdx
  __int64 v20; // r11
  int v21; // edx
  int v22; // r8d
  int v23; // [rsp+14h] [rbp-CCh]
  __int64 v24; // [rsp+20h] [rbp-C0h]
  unsigned __int8 v27; // [rsp+38h] [rbp-A8h]
  int v28; // [rsp+4Ch] [rbp-94h] BYREF
  unsigned __int64 v29[2]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v30[64]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v31; // [rsp+A0h] [rbp-40h]
  char v32; // [rsp+A8h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 64);
  v32 = 0;
  v6 = *a3;
  v29[0] = (unsigned __int64)v30;
  v29[1] = 0x400000000LL;
  v31 = 0;
  v7 = *(__int64 **)(v5 + 24 * v6 + 8);
  if ( v7 && *((_BYTE *)v7 + 8) )
  {
    do
    {
      v8 = v7;
      v7 = (__int64 *)*v7;
    }
    while ( v7 && *((_BYTE *)v7 + 8) );
    result = sub_1371320(a1, a2, v8, v29);
    if ( !(_BYTE)result )
      goto LABEL_8;
LABEL_7:
    sub_1373530(a1, a3, a2, v29);
    result = 1;
    goto LABEL_8;
  }
  v24 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v6);
  v10 = sub_157EBA0(v24);
  v11 = v10;
  if ( !v10 )
    goto LABEL_7;
  v23 = sub_15F4D60(v10);
  if ( !v23 )
    goto LABEL_7;
  v12 = 0;
  while ( 1 )
  {
    v3 = v12 | v3 & 0xFFFFFFFF00000000LL;
    sub_13774A0(*(_QWORD *)(a1 + 112), v24, v11, v3);
    v14 = sub_15F4DF0(v11, v12);
    v15 = *(_DWORD *)(a1 + 184);
    v13 = -1;
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 168);
      v18 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( v14 == *v19 )
      {
LABEL_14:
        v13 = *((_DWORD *)v19 + 2);
      }
      else
      {
        v21 = 1;
        while ( v20 != -8 )
        {
          v22 = v21 + 1;
          v18 = v16 & (v21 + v18);
          v19 = (__int64 *)(v17 + 16LL * v18);
          v20 = *v19;
          if ( v14 == *v19 )
            goto LABEL_14;
          v21 = v22;
        }
        v13 = -1;
      }
    }
    v28 = v13;
    result = sub_13710E0(a1, v29, a2, a3, &v28);
    if ( !(_BYTE)result )
      break;
    if ( v23 == ++v12 )
      goto LABEL_7;
  }
LABEL_8:
  if ( (_BYTE *)v29[0] != v30 )
  {
    v27 = result;
    _libc_free(v29[0]);
    return v27;
  }
  return result;
}
