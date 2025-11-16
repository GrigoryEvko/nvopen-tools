// Function: sub_305F6F0
// Address: 0x305f6f0
//
_BYTE *__fastcall sub_305F6F0(__int64 a1, int *a2)
{
  const char *v2; // r15
  size_t v3; // r12
  const char **v4; // rbx
  int v5; // eax
  int v6; // eax
  _BYTE *result; // rax
  int v8; // eax
  unsigned int v9; // r8d
  __int64 *v10; // rcx
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 *v13; // rcx
  __int64 v14; // r13
  __int64 *v15; // [rsp+0h] [rbp-90h]
  unsigned int v16; // [rsp+8h] [rbp-88h]
  int v17; // [rsp+Ch] [rbp-84h]
  _QWORD v18[2]; // [rsp+30h] [rbp-60h] BYREF
  int v19; // [rsp+40h] [rbp-50h]
  const char *v20; // [rsp+48h] [rbp-48h]
  __int64 v21; // [rsp+50h] [rbp-40h]
  int v22; // [rsp+58h] [rbp-38h]
  _BYTE v23[48]; // [rsp+60h] [rbp-30h] BYREF

  v2 = "__CUDA_ARCH";
  v3 = 11;
  v4 = (const char **)v18;
  v5 = *a2;
  v18[0] = "__CUDA_ARCH";
  v18[1] = 11;
  v17 = v5;
  v19 = v5;
  v20 = "__CUDA_FTZ";
  v6 = *((unsigned __int8 *)a2 + 4);
  v21 = 10;
  v22 = v6;
  sub_C926D0(a1, 2, 16);
  while ( 1 )
  {
    v8 = sub_C92610();
    v9 = sub_C92740(a1, v2, v3, v8);
    v10 = (__int64 *)(*(_QWORD *)a1 + 8LL * v9);
    if ( !*v10 )
      break;
    if ( *v10 == -8 )
    {
      --*(_DWORD *)(a1 + 16);
      break;
    }
    v4 += 3;
    result = v23;
    if ( v4 == (const char **)v23 )
      return result;
LABEL_3:
    v3 = (size_t)v4[1];
    v2 = *v4;
    v17 = *((_DWORD *)v4 + 4);
  }
  v15 = v10;
  v16 = v9;
  v11 = sub_C7D670(v3 + 17, 8);
  v12 = v16;
  v13 = v15;
  v14 = v11;
  if ( v3 )
  {
    memcpy((void *)(v11 + 16), v2, v3);
    v12 = v16;
    v13 = v15;
  }
  v4 += 3;
  *(_BYTE *)(v14 + v3 + 16) = 0;
  *(_QWORD *)v14 = v3;
  *(_DWORD *)(v14 + 8) = v17;
  *v13 = v14;
  ++*(_DWORD *)(a1 + 12);
  sub_C929D0((__int64 *)a1, v12);
  result = v23;
  if ( v4 != (const char **)v23 )
    goto LABEL_3;
  return result;
}
